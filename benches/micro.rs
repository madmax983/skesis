#![allow(dead_code)] // Struct fields exist for realistic layout/size, not read access.

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use skesis::World;

#[derive(Clone, Copy)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy)]
struct Velocity {
    dx: f32,
    dy: f32,
}

#[derive(Clone, Copy)]
struct Health {
    hp: f32,
    max_hp: f32,
}

#[derive(Clone, Copy)]
struct Tag;

// Extra zero-size tag components to generate many archetypes.
#[derive(Clone, Copy)]
struct TagA;
#[derive(Clone, Copy)]
struct TagB;
#[derive(Clone, Copy)]
struct TagC;
#[derive(Clone, Copy)]
struct TagD;
#[derive(Clone, Copy)]
struct TagE;
#[derive(Clone, Copy)]
struct TagF;

const ENTITY_COUNT: usize = 10_000;

// ── Query iteration (the original benchmark) ─────────────────────────

fn bench_query_pair(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_pair");
    group.throughput(Throughput::Elements(ENTITY_COUNT as u64));

    let mut world = World::new();
    for i in 0..ENTITY_COUNT {
        let entity = world.spawn_empty();
        world.add_component(
            entity,
            Position {
                x: i as f32,
                y: (i % 100) as f32,
            },
        );
        world.add_component(entity, Velocity { dx: 1.0, dy: 1.0 });
    }

    let plan = world.plan_query_pair::<Position, Velocity>();

    group.bench_function("cached", |b| {
        b.iter(|| {
            let mut checksum = 0.0f32;
            world.for_each_pair_with_plan(&plan, |entity, pos, vel| {
                checksum += entity.index() as f32 + pos.x + vel.dx;
            });
            black_box(checksum);
        });
    });

    group.bench_function("uncached", |b| {
        b.iter(|| {
            let mut checksum = 0.0f32;
            world.for_each_pair::<Position, Velocity>(|entity, pos, vel| {
                checksum += entity.index() as f32 + pos.x + vel.dx;
            });
            black_box(checksum);
        });
    });

    group.finish();
}

// ── Query scaling (varies entity count) ──────────────────────────────

fn bench_query_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_scaling");

    for &count in &[100, 1_000, 10_000, 100_000] {
        group.throughput(Throughput::Elements(count as u64));

        let mut world = World::new();
        for i in 0..count {
            let entity = world.spawn_empty();
            world.add_component(
                entity,
                Position {
                    x: i as f32,
                    y: 0.0,
                },
            );
            world.add_component(entity, Velocity { dx: 1.0, dy: 1.0 });
        }

        let plan = world.plan_query_pair::<Position, Velocity>();

        group.bench_function(format!("n={count}"), |b| {
            b.iter(|| {
                let mut checksum = 0.0f32;
                world.for_each_pair_with_plan(&plan, |entity, pos, vel| {
                    checksum += entity.index() as f32 + pos.x + vel.dx;
                });
                black_box(checksum);
            });
        });
    }

    group.finish();
}

// ── Archetype fragmentation (many small archetypes) ──────────────────

fn bench_fragmented_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("fragmented_query");
    group.throughput(Throughput::Elements(ENTITY_COUNT as u64));

    // Create 10K entities spread across many archetypes by varying
    // which extra components they have.
    let mut world = World::new();
    for i in 0..ENTITY_COUNT {
        let entity = world.spawn_empty();
        world.add_component(
            entity,
            Position {
                x: i as f32,
                y: 0.0,
            },
        );
        world.add_component(entity, Velocity { dx: 1.0, dy: 1.0 });

        // 50% also have Health
        if i % 2 == 0 {
            world.add_component(
                entity,
                Health {
                    hp: 100.0,
                    max_hp: 100.0,
                },
            );
        }
        // 25% also have Tag
        if i % 4 == 0 {
            world.add_component(entity, Tag);
        }
    }

    let arch_count = world.archetype_count();

    group.bench_function(format!("uncached_{arch_count}_archetypes"), |b| {
        b.iter(|| {
            let mut checksum = 0.0f32;
            world.for_each_pair::<Position, Velocity>(|entity, pos, vel| {
                checksum += entity.index() as f32 + pos.x + vel.dx;
            });
            black_box(checksum);
        });
    });

    let plan = world.plan_query_pair::<Position, Velocity>();

    group.bench_function(format!("cached_{arch_count}_archetypes"), |b| {
        b.iter(|| {
            let mut checksum = 0.0f32;
            world.for_each_pair_with_plan(&plan, |entity, pos, vel| {
                checksum += entity.index() as f32 + pos.x + vel.dx;
            });
            black_box(checksum);
        });
    });

    group.finish();
}

// ── Heavy fragmentation (64+ archetypes) ─────────────────────────────

fn bench_heavy_fragmentation(c: &mut Criterion) {
    let mut group = c.benchmark_group("heavy_fragmentation");
    group.throughput(Throughput::Elements(ENTITY_COUNT as u64));

    // 6 independent tag bits = up to 2^6 = 64 archetype combinations.
    // All entities have Position + Velocity; tags vary.
    let mut world = World::new();
    for i in 0..ENTITY_COUNT {
        let entity = world.spawn_empty();
        world.add_component(
            entity,
            Position {
                x: i as f32,
                y: 0.0,
            },
        );
        world.add_component(entity, Velocity { dx: 1.0, dy: 1.0 });

        if i & 1 != 0 {
            world.add_component(entity, TagA);
        }
        if i & 2 != 0 {
            world.add_component(entity, TagB);
        }
        if i & 4 != 0 {
            world.add_component(entity, TagC);
        }
        if i & 8 != 0 {
            world.add_component(entity, TagD);
        }
        if i & 16 != 0 {
            world.add_component(entity, TagE);
        }
        if i & 32 != 0 {
            world.add_component(entity, TagF);
        }
    }

    let arch_count = world.archetype_count();

    group.bench_function(format!("uncached_{arch_count}_archetypes"), |b| {
        b.iter(|| {
            let mut checksum = 0.0f32;
            world.for_each_pair::<Position, Velocity>(|entity, pos, vel| {
                checksum += entity.index() as f32 + pos.x + vel.dx;
            });
            black_box(checksum);
        });
    });

    let plan = world.plan_query_pair::<Position, Velocity>();

    group.bench_function(format!("cached_{arch_count}_archetypes"), |b| {
        b.iter(|| {
            let mut checksum = 0.0f32;
            world.for_each_pair_with_plan(&plan, |entity, pos, vel| {
                checksum += entity.index() as f32 + pos.x + vel.dx;
            });
            black_box(checksum);
        });
    });

    group.finish();
}

// ── Spawn + add_component (entity creation hot path) ─────────────────

fn bench_spawn(c: &mut Criterion) {
    let mut group = c.benchmark_group("spawn");
    group.throughput(Throughput::Elements(ENTITY_COUNT as u64));

    group.bench_function("spawn_empty", |b| {
        b.iter(|| {
            let mut world = World::new();
            for _ in 0..ENTITY_COUNT {
                black_box(world.spawn_empty());
            }
        });
    });

    group.bench_function("spawn_with_2_components", |b| {
        b.iter(|| {
            let mut world = World::new();
            for i in 0..ENTITY_COUNT {
                let entity = world.spawn_empty();
                world.add_component(
                    entity,
                    Position {
                        x: i as f32,
                        y: 0.0,
                    },
                );
                world.add_component(entity, Velocity { dx: 1.0, dy: 1.0 });
            }
        });
    });

    group.finish();
}

// ── Batch spawn (spawn_with vs spawn_empty + add_component) ──────────

fn bench_spawn_with(c: &mut Criterion) {
    let mut group = c.benchmark_group("spawn_with");
    group.throughput(Throughput::Elements(ENTITY_COUNT as u64));

    group.bench_function("add_component_chain", |b| {
        b.iter(|| {
            let mut world = World::new();
            for i in 0..ENTITY_COUNT {
                let entity = world.spawn_empty();
                world.add_component(
                    entity,
                    Position {
                        x: i as f32,
                        y: 0.0,
                    },
                );
                world.add_component(entity, Velocity { dx: 1.0, dy: 1.0 });
            }
            black_box(&world);
        });
    });

    group.bench_function("spawn_with_bundle", |b| {
        b.iter(|| {
            let mut world = World::new();
            for i in 0..ENTITY_COUNT {
                let _entity = world.spawn_with((
                    Position {
                        x: i as f32,
                        y: 0.0,
                    },
                    Velocity { dx: 1.0, dy: 1.0 },
                ));
            }
            black_box(&world);
        });
    });

    group.bench_function("spawn_with_3_components", |b| {
        b.iter(|| {
            let mut world = World::new();
            for i in 0..ENTITY_COUNT {
                let _entity = world.spawn_with((
                    Position {
                        x: i as f32,
                        y: 0.0,
                    },
                    Velocity { dx: 1.0, dy: 1.0 },
                    Health {
                        hp: 100.0,
                        max_hp: 100.0,
                    },
                ));
            }
            black_box(&world);
        });
    });

    group.finish();
}

// ── Mutable query iteration ──────────────────────────────────────────

fn bench_query_mut(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_mut");
    group.throughput(Throughput::Elements(ENTITY_COUNT as u64));

    let mut world = World::new();
    for i in 0..ENTITY_COUNT {
        let entity = world.spawn_empty();
        world.add_component(
            entity,
            Position {
                x: i as f32,
                y: 0.0,
            },
        );
        world.add_component(entity, Velocity { dx: 1.0, dy: 1.0 });
    }

    group.bench_function("for_each_mut", |b| {
        b.iter(|| {
            world.for_each_mut::<Position>(|_entity, pos| {
                pos.x += 1.0;
                pos.y += 1.0;
            });
        });
    });

    let plan = world.plan_query_mut::<Position>();

    group.bench_function("for_each_mut_with_plan", |b| {
        b.iter(|| {
            world.for_each_mut_with_plan(&plan, |_entity, pos| {
                pos.x += 1.0;
                pos.y += 1.0;
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_query_pair,
    bench_query_scaling,
    bench_fragmented_query,
    bench_heavy_fragmentation,
    bench_spawn,
    bench_spawn_with,
    bench_query_mut,
);
criterion_main!(benches);
