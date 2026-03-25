use std::any::Any;

pub(crate) type ComponentValue = Box<dyn Any + Send + Sync>;
pub(crate) type ColumnFactory = fn() -> Box<dyn ErasedColumn>;

pub(crate) trait ErasedColumn: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn len(&self) -> usize;
    fn push_boxed(&mut self, value: ComponentValue);
    fn swap_remove_into(&mut self, index: usize, destination: &mut dyn ErasedColumn);
    fn swap_remove_boxed(&mut self, index: usize) -> ComponentValue;
    fn swap_remove_drop(&mut self, index: usize);
    fn as_bytes(&self) -> &[u8];
    fn snapshot_bytes(&self) -> Vec<u8>;
    fn element_stride(&self) -> usize;
    /// Restore column contents from raw bytes. Replaces all existing data.
    fn restore_from_bytes(&mut self, bytes: &[u8]);
    /// Clear all elements.
    fn clear(&mut self);
}

pub(crate) struct TypedColumn<T> {
    data: Vec<T>,
}

impl<T> TypedColumn<T> {
    pub(crate) fn new() -> Self {
        Self { data: Vec::new() }
    }

    #[inline(always)]
    pub(crate) fn push(&mut self, value: T) {
        self.data.push(value);
    }

    pub(crate) fn swap_remove(&mut self, index: usize) -> T {
        self.data.swap_remove(index)
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        let ptr = self.data.as_ptr().cast::<u8>();
        let len = self.data.len() * std::mem::size_of::<T>();
        // SAFETY: Vec<T> backing storage is contiguous and aligned.
        // Reinterpreting as bytes is always safe for reads.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    pub(crate) fn snapshot_bytes(&self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }

    pub(crate) fn element_stride(&self) -> usize {
        std::mem::size_of::<T>()
    }
}

impl<T: 'static + Send + Sync> ErasedColumn for TypedColumn<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn push_boxed(&mut self, value: ComponentValue) {
        match value.downcast::<T>() {
            Ok(typed) => self.data.push(*typed),
            Err(_) => panic!("typed column received mismatched component type"),
        }
    }

    fn swap_remove_into(&mut self, index: usize, destination: &mut dyn ErasedColumn) {
        let destination_typed = destination
            .as_any_mut()
            .downcast_mut::<TypedColumn<T>>()
            .expect("typed column destination mismatch while moving component row");
        destination_typed.data.push(self.data.swap_remove(index));
    }

    fn swap_remove_boxed(&mut self, index: usize) -> ComponentValue {
        Box::new(self.data.swap_remove(index))
    }

    fn swap_remove_drop(&mut self, index: usize) {
        self.data.swap_remove(index);
    }

    fn as_bytes(&self) -> &[u8] {
        TypedColumn::as_bytes(self)
    }

    fn snapshot_bytes(&self) -> Vec<u8> {
        TypedColumn::snapshot_bytes(self)
    }

    fn element_stride(&self) -> usize {
        TypedColumn::element_stride(self)
    }

    fn restore_from_bytes(&mut self, bytes: &[u8]) {
        let stride = std::mem::size_of::<T>();
        assert!(
            stride > 0 && bytes.len() % stride == 0,
            "byte length must be a multiple of element stride"
        );
        let count = bytes.len() / stride;
        self.data.clear();
        self.data.reserve(count);
        // SAFETY: We copy raw bytes into the Vec's backing storage.
        // Caller guarantees bytes came from a compatible TypedColumn<T>.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.data.as_mut_ptr().cast::<u8>(),
                bytes.len(),
            );
            self.data.set_len(count);
        }
    }

    fn clear(&mut self) {
        self.data.clear();
    }
}

pub(crate) struct BoxedColumn {
    values: Vec<ComponentValue>,
}

impl BoxedColumn {
    pub(crate) fn new() -> Self {
        Self { values: Vec::new() }
    }
}

impl ErasedColumn for BoxedColumn {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn push_boxed(&mut self, value: ComponentValue) {
        self.values.push(value);
    }

    fn swap_remove_into(&mut self, index: usize, destination: &mut dyn ErasedColumn) {
        let destination_boxed = destination
            .as_any_mut()
            .downcast_mut::<BoxedColumn>()
            .expect("boxed column destination mismatch while moving component row");
        destination_boxed
            .values
            .push(self.values.swap_remove(index));
    }

    fn swap_remove_boxed(&mut self, index: usize) -> ComponentValue {
        self.values.swap_remove(index)
    }

    fn swap_remove_drop(&mut self, index: usize) {
        let _ = self.values.swap_remove(index);
    }

    fn as_bytes(&self) -> &[u8] {
        // BoxedColumn stores trait objects on the heap; no contiguous byte access is possible.
        &[]
    }

    fn snapshot_bytes(&self) -> Vec<u8> {
        Vec::new()
    }

    fn element_stride(&self) -> usize {
        0
    }

    fn restore_from_bytes(&mut self, _bytes: &[u8]) {
        // BoxedColumn cannot be restored from raw bytes.
    }

    fn clear(&mut self) {
        self.values.clear();
    }
}

pub(crate) fn typed_column_factory<T: 'static + Send + Sync>() -> Box<dyn ErasedColumn> {
    Box::new(TypedColumn::<T>::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_column_as_bytes_roundtrip() {
        let mut col = TypedColumn::<f32>::new();
        col.push(1.0);
        col.push(2.0);
        col.push(3.0);

        let bytes = col.as_bytes();
        assert_eq!(bytes.len(), 3 * std::mem::size_of::<f32>());

        let snapshot = col.snapshot_bytes();
        col.push(4.0); // mutate after snapshot
        assert_ne!(col.as_bytes().len(), snapshot.len());
    }

    #[test]
    fn typed_column_element_stride() {
        let col = TypedColumn::<[f32; 2]>::new();
        assert_eq!(col.element_stride(), std::mem::size_of::<[f32; 2]>());
    }
}
