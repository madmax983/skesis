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

const ENTITY_COUNT: usize = 10_000;

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

criterion_group!(benches, bench_query_pair);
criterion_main!(benches);
