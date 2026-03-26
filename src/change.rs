//! Anchor+delta change detection for ECS columns.

use std::any::TypeId;
use std::collections::HashMap;

/// A contiguous run of changed bytes at a given offset.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ByteRun {
    pub offset: usize,
    pub len: usize,
}

/// Compare two equal-length byte slices, returning runs of differing bytes.
#[cfg(test)]
pub(crate) fn diff_bytes(before: &[u8], after: &[u8]) -> Vec<ByteRun> {
    debug_assert_eq!(before.len(), after.len());
    let mut runs = Vec::new();
    let len = before.len();
    let mut i = 0;

    while i < len {
        if before[i] == after[i] {
            i += 1;
        } else {
            let start = i;
            while i < len && before[i] != after[i] {
                i += 1;
            }
            runs.push(ByteRun {
                offset: start,
                len: i - start,
            });
        }
    }

    runs
}

/// Return true if any byte differs between two equal-length slices.
pub(crate) fn any_bytes_differ(before: &[u8], after: &[u8]) -> bool {
    debug_assert_eq!(before.len(), after.len());
    let len = before.len();

    // Use unaligned u64 reads for the bulk comparison.
    let word_count = len / 8;
    if word_count > 0 {
        let before_ptr = before.as_ptr();
        let after_ptr = after.as_ptr();
        for i in 0..word_count {
            let offset = i * 8;
            // SAFETY: offset + 8 <= word_count * 8 <= len, so reads are in bounds.
            // read_unaligned handles any alignment.
            let bw = unsafe { before_ptr.add(offset).cast::<u64>().read_unaligned() };
            let aw = unsafe { after_ptr.add(offset).cast::<u64>().read_unaligned() };
            if bw != aw {
                return true;
            }
        }
    }

    // Byte-level tail comparison.
    let tail_start = word_count * 8;
    before[tail_start..].iter().zip(after[tail_start..].iter()).any(|(a, b)| a != b)
}

/// Build a bitset (one bit per element of size `stride`) marking which elements differ.
/// Returns a Vec<u64> where bit N is set if element N has at least one changed byte.
pub fn diff_to_bitset(before: &[u8], after: &[u8], stride: usize) -> Vec<u64> {
    debug_assert_eq!(before.len(), after.len());
    let element_count = if stride == 0 {
        0
    } else {
        before.len() / stride
    };
    let word_count = element_count.div_ceil(64);
    let mut bits = vec![0u64; word_count];

    #[cfg(target_arch = "x86_64")]
    {
        if stride == 8 && is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected, pointers are valid for the slice lengths.
            unsafe {
                diff_to_bitset_avx2_stride8(before, after, element_count, &mut bits);
            }
            return bits;
        }
    }

    // Scalar fast path: when stride is a multiple of 8, compare u64 words directly.
    if stride >= 8 && stride.is_multiple_of(8) {
        let words_per_element = stride / 8;
        let before_words =
            unsafe { std::slice::from_raw_parts(before.as_ptr().cast::<u64>(), before.len() / 8) };
        let after_words =
            unsafe { std::slice::from_raw_parts(after.as_ptr().cast::<u64>(), after.len() / 8) };

        for i in 0..element_count {
            let base = i * words_per_element;
            let mut changed = false;
            for w in 0..words_per_element {
                if before_words[base + w] != after_words[base + w] {
                    changed = true;
                    break;
                }
            }
            if changed {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }
    } else {
        for i in 0..element_count {
            let start = i * stride;
            let end = start + stride;
            if any_bytes_differ(&before[start..end], &after[start..end]) {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }
    }

    bits
}

/// AVX2 fast path for stride=8: compare 4 elements (32 bytes) per iteration.
///
/// Uses vpxor + comparison to produce a 4-bit mask per chunk, then scatters
/// the changed bits into the output bitset.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn diff_to_bitset_avx2_stride8(
    before: &[u8],
    after: &[u8],
    element_count: usize,
    bits: &mut [u64],
) {
    use std::arch::x86_64::{
        __m256i, _mm256_castsi256_pd, _mm256_cmpeq_epi64, _mm256_loadu_si256,
        _mm256_movemask_pd, _mm256_setzero_si256, _mm256_xor_si256,
    };

    // SAFETY: Caller guarantees valid aligned slices of sufficient length,
    // and the #[target_feature(enable = "avx2")] gate ensures AVX2 is available.
    unsafe {
        let zero = _mm256_setzero_si256();
        let chunks = element_count / 4;
        let before_ptr = before.as_ptr();
        let after_ptr = after.as_ptr();

        for chunk in 0..chunks {
            let offset = chunk * 32; // 4 elements × 8 bytes
            let b = _mm256_loadu_si256(before_ptr.add(offset).cast::<__m256i>());
            let a = _mm256_loadu_si256(after_ptr.add(offset).cast::<__m256i>());

            // XOR: zero lanes where equal, non-zero where different.
            let xor = _mm256_xor_si256(b, a);
            // Compare each 64-bit lane to zero: all-ones if equal (unchanged).
            let eq = _mm256_cmpeq_epi64(xor, zero);
            // Extract MSB of each 64-bit lane as a 4-bit mask.
            // Bit is 1 if lane was all-ones (equal/unchanged), 0 if different.
            let mask = _mm256_movemask_pd(_mm256_castsi256_pd(eq)) as u32;
            // Invert: we want bits set where elements CHANGED (were not equal).
            let changed_mask = (!mask) & 0xF;

            if changed_mask != 0 {
                let global_element = chunk * 4;
                let word_idx = global_element / 64;
                let bit_offset = global_element % 64;
                // The 4 changed bits land within one or two u64 words.
                bits[word_idx] |= (changed_mask as u64) << bit_offset;
                // Handle overflow into next word (when bit_offset > 60).
                if bit_offset > 60 && word_idx + 1 < bits.len() {
                    bits[word_idx + 1] |= (changed_mask as u64) >> (64 - bit_offset);
                }
            }
        }

        // Scalar tail: remaining elements that don't fill a full 4-element chunk.
        let tail_start = chunks * 4;
        let before_words =
            std::slice::from_raw_parts(before.as_ptr().cast::<u64>(), before.len() / 8);
        let after_words =
            std::slice::from_raw_parts(after.as_ptr().cast::<u64>(), after.len() / 8);

        for i in tail_start..element_count {
            if before_words[i] != after_words[i] {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }
    }
}

/// Tracks changes to a component column using anchor+delta snapshots.
#[derive(Default)]
pub(crate) struct ChangeTracker {
    /// Bumped whenever the column may have been mutated.
    pub column_version: u64,
    /// Per-reader snapshots: reader_id -> (version_at_snapshot, frozen_bytes).
    snapshots: HashMap<u64, (u64, Vec<u8>)>,
}

impl ChangeTracker {
    pub(crate) fn bump_version(&mut self) {
        self.column_version = self.column_version.wrapping_add(1);
    }

    pub(crate) fn take_snapshot(&mut self, reader_id: u64, column_bytes: &[u8]) {
        self.snapshots
            .insert(reader_id, (self.column_version, column_bytes.to_vec()));
    }

    #[allow(dead_code)]
    pub(crate) fn has_changes(&self, reader_id: u64, current_bytes: &[u8]) -> bool {
        let Some((snap_version, snap_bytes)) = self.snapshots.get(&reader_id) else {
            return true; // No snapshot = assume everything changed.
        };
        if *snap_version == self.column_version {
            return false; // Fast exit: version unchanged.
        }
        any_bytes_differ(snap_bytes, current_bytes)
    }

    pub(crate) fn snapshots_contains(&self, reader_id: u64) -> bool {
        self.snapshots.contains_key(&reader_id)
    }

    /// Get the snapshot version for a reader. Returns None if no snapshot exists.
    pub(crate) fn snapshot_version(&self, reader_id: u64) -> Option<u64> {
        self.snapshots.get(&reader_id).map(|(v, _)| *v)
    }

    /// Get the raw snapshot bytes for a reader. Returns None if no snapshot exists.
    pub(crate) fn snapshot_bytes(&self, reader_id: u64) -> Option<&[u8]> {
        self.snapshots.get(&reader_id).map(|(_, b)| b.as_slice())
    }

    /// Returns None if no changes (fast exit). Some(bitset) with per-element changed bits otherwise.
    #[allow(dead_code)]
    pub(crate) fn changed_bitset(
        &self,
        reader_id: u64,
        current_bytes: &[u8],
        element_stride: usize,
    ) -> Option<Vec<u64>> {
        let Some((snap_version, snap_bytes)) = self.snapshots.get(&reader_id) else {
            return None; // No snapshot = caller should treat all as changed.
        };
        if *snap_version == self.column_version {
            return None; // Fast exit: nothing mutated since snapshot.
        }
        // If column size changed (entity added/removed), mark all current elements as changed.
        if snap_bytes.len() != current_bytes.len() {
            let element_count = if element_stride == 0 {
                0
            } else {
                current_bytes.len() / element_stride
            };
            let word_count = element_count.div_ceil(64);
            let mut bits = vec![u64::MAX; word_count];
            // Mask the final word so only valid element bits are set.
            let remainder = element_count % 64;
            if remainder != 0 && word_count > 0 {
                bits[word_count - 1] = (1u64 << remainder) - 1;
            }
            return Some(bits);
        }
        Some(diff_to_bitset(snap_bytes, current_bytes, element_stride))
    }
}

/// Standalone historical storage for change detection.
///
/// Separated from World so the hot path (archetype iteration) never touches
/// historical data. This struct can be serialized independently for networking
/// or replay without touching the live world.
///
/// **Architecture (mirrors AletheiaDB):**
/// - Anchors = live `TypedColumn<T>` data in archetypes (current storage, hot path)
/// - Snapshots = `Vec<u8>` in this struct (historical storage, cold path)
///
/// The two never share cache lines during iteration.
pub struct ChangeHistory {
    trackers: HashMap<TypeId, ChangeTracker>,
}

impl ChangeHistory {
    /// Create an empty change history.
    pub fn new() -> Self {
        Self {
            trackers: HashMap::new(),
        }
    }

    /// Register a component type for change tracking.
    pub fn track<T: 'static>(&mut self) {
        self.trackers
            .entry(TypeId::of::<T>())
            .or_default();
    }

    /// Check if a component type is tracked.
    pub fn is_tracked<T: 'static>(&self) -> bool {
        self.trackers.contains_key(&TypeId::of::<T>())
    }

    /// Bump the version counter for a tracked component type.
    pub fn bump_version<T: 'static>(&mut self) {
        if let Some(tracker) = self.trackers.get_mut(&TypeId::of::<T>()) {
            tracker.bump_version();
        }
    }

    /// Bump the version counter by TypeId (for internal use).
    pub(crate) fn bump_version_by_type_id(&mut self, type_id: &TypeId) {
        if let Some(tracker) = self.trackers.get_mut(type_id) {
            tracker.bump_version();
        }
    }

    /// Take a snapshot of column bytes for a reader.
    pub fn take_snapshot<T: 'static>(&mut self, reader_id: u64, column_bytes: &[u8]) {
        if let Some(tracker) = self.trackers.get_mut(&TypeId::of::<T>()) {
            tracker.take_snapshot(reader_id, column_bytes);
        }
    }

    /// Get the tracker for a component type (for query methods).
    pub(crate) fn tracker<T: 'static>(&self) -> Option<&ChangeTracker> {
        self.trackers.get(&TypeId::of::<T>())
    }

}

impl Default for ChangeHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_slices_no_runs() {
        let a = [1u8, 2, 3, 4];
        assert!(diff_bytes(&a, &a).is_empty());
    }

    #[test]
    fn single_byte_change() {
        let before = [0u8, 0, 0, 0];
        let after = [0u8, 0, 42, 0];
        let runs = diff_bytes(&before, &after);
        assert_eq!(runs, vec![ByteRun { offset: 2, len: 1 }]);
    }

    #[test]
    fn adjacent_changes_merged() {
        let before = [0u8; 5];
        let after = [0u8, 1, 2, 3, 0];
        let runs = diff_bytes(&before, &after);
        assert_eq!(runs, vec![ByteRun { offset: 1, len: 3 }]);
    }

    #[test]
    fn any_bytes_differ_true() {
        let a = [0u8; 4];
        let b = [0u8, 0, 1, 0];
        assert!(any_bytes_differ(&a, &b));
    }

    #[test]
    fn any_bytes_differ_false() {
        let a = [1u8, 2, 3];
        assert!(!any_bytes_differ(&a, &a));
    }

    #[test]
    fn diff_to_bitset_marks_changed_elements() {
        // 4 elements of stride 2 = 8 bytes total.
        let before = [0u8, 0, 0, 0, 0, 0, 0, 0];
        let after = [0u8, 0, 1, 2, 0, 0, 3, 4]; // elements 1 and 3 changed
        let bits = diff_to_bitset(&before, &after, 2);
        assert_eq!(bits.len(), 1); // 4 elements fits in one u64
        assert_eq!(bits[0] & (1 << 1), 1 << 1); // element 1 changed
        assert_eq!(bits[0] & (1 << 3), 1 << 3); // element 3 changed
        assert_eq!(bits[0] & (1 << 0), 0); // element 0 unchanged
        assert_eq!(bits[0] & (1 << 2), 0); // element 2 unchanged
    }

    #[test]
    fn diff_to_bitset_stride8_hits_avx2_path() {
        // 8 elements of stride 8 = 64 bytes (two AVX2 chunks of 4 elements).
        let before = [0u8; 64];
        let mut after = [0u8; 64];
        // Change element 1 (bytes 8..16) and element 5 (bytes 40..48).
        after[8] = 1;
        after[40] = 1;

        let bits = diff_to_bitset(&before, &after, 8);
        assert_eq!(bits.len(), 1);
        assert_eq!(bits[0] & (1 << 0), 0); // element 0 unchanged
        assert_eq!(bits[0] & (1 << 1), 1 << 1); // element 1 changed
        assert_eq!(bits[0] & (1 << 2), 0); // element 2 unchanged
        assert_eq!(bits[0] & (1 << 5), 1 << 5); // element 5 changed
        assert_eq!(bits[0] & (1 << 7), 0); // element 7 unchanged
    }

    #[test]
    fn diff_to_bitset_stride8_large() {
        // 100 elements of stride 8 = 800 bytes. Tests AVX2 + scalar tail.
        let before = vec![0u8; 800];
        let mut after = vec![0u8; 800];
        // Change elements 0, 50, 99.
        after[0] = 1; // element 0
        after[400] = 1; // element 50
        after[792] = 1; // element 99

        let bits = diff_to_bitset(&before, &after, 8);
        assert_eq!(bits.len(), 2); // 100 elements needs 2 u64 words
        assert_ne!(bits[0] & (1 << 0), 0); // element 0
        assert_ne!(bits[0] & (1 << 50), 0); // element 50
        assert_ne!(bits[1] & (1 << (99 - 64)), 0); // element 99
        // Spot check unchanged.
        assert_eq!(bits[0] & (1 << 1), 0);
        assert_eq!(bits[0] & (1 << 49), 0);
    }

    #[test]
    fn tracker_snapshot_and_check_no_change() {
        let mut tracker = ChangeTracker::default();
        let column_bytes = [0u8; 16];
        tracker.take_snapshot(1, &column_bytes);
        let changed = tracker.has_changes(1, &column_bytes);
        assert!(!changed);
    }

    #[test]
    fn tracker_detects_change_after_version_bump() {
        let mut tracker = ChangeTracker::default();
        let before = [0u8; 16];
        tracker.take_snapshot(1, &before);
        tracker.bump_version();
        let after = [0u8, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let changed = tracker.has_changes(1, &after);
        assert!(changed);
    }

    #[test]
    fn tracker_changed_bitset() {
        let mut tracker = ChangeTracker::default();
        let before = [0u8; 16];
        tracker.take_snapshot(1, &before);
        tracker.bump_version();
        let mut after = [0u8; 16];
        after[4] = 1;
        after[12] = 1;
        let bits = tracker.changed_bitset(1, &after, 4);
        assert!(bits.is_some());
        let bits = bits.unwrap();
        assert_eq!(bits[0] & (1 << 1), 1 << 1);
        assert_eq!(bits[0] & (1 << 3), 1 << 3);
        assert_eq!(bits[0] & (1 << 0), 0);
    }

    #[test]
    fn tracker_fast_exit_when_version_unchanged() {
        let mut tracker = ChangeTracker::default();
        let bytes = [0u8; 16];
        tracker.take_snapshot(1, &bytes);
        let bits = tracker.changed_bitset(1, &bytes, 4);
        assert!(bits.is_none());
    }

    #[test]
    fn tracker_no_snapshot_has_changes_returns_true() {
        let tracker = ChangeTracker::default();
        let bytes = [0u8; 16];
        assert!(tracker.has_changes(99, &bytes));
    }

    #[test]
    fn tracker_per_reader_independent() {
        let mut tracker = ChangeTracker::default();
        let bytes = [0u8; 8];
        tracker.take_snapshot(1, &bytes);
        tracker.take_snapshot(2, &bytes);
        tracker.bump_version();
        let mut changed = [0u8; 8];
        changed[0] = 1;
        // Reader 1 sees changes.
        assert!(tracker.has_changes(1, &changed));
        // Reader 2 also sees changes (independent).
        assert!(tracker.has_changes(2, &changed));
        // Re-snapshot for reader 1 only.
        tracker.take_snapshot(1, &changed);
        // Reader 1 no longer sees changes.
        assert!(!tracker.has_changes(1, &changed));
        // Reader 2 still sees changes.
        assert!(tracker.has_changes(2, &changed));
    }
}
