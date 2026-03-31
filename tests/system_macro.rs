//! Integration tests for the #[system] proc macro.

use skesis::{system, Res, ResMut, Query, World};

struct Score(u32);
struct DeltaTime(f32);
struct Position { x: f32, y: f32 }
struct Velocity { dx: f32, dy: f32 }

// --- Test: Res + ResMut ---

#[system]
fn increment_score(mut score: ResMut<Score>, dt: Res<DeltaTime>) {
    score.0 += (dt.0 * 100.0) as u32;
}

#[test]
fn test_res_resmut_system() {
    let mut world = World::new();
    world.insert_resource(Score(0));
    world.insert_resource(DeltaTime(0.016));

    let mut state = increment_score_init(&mut world);
    increment_score_run(&mut world, &mut state);

    assert_eq!(world.get_resource::<Score>().unwrap().0, 1);
}

// --- Test: Query with single component ---

#[system]
fn count_positions(query: Query<&Position>, mut score: ResMut<Score>) {
    let mut count = 0u32;
    for (_entity, _pos) in &query {
        count += 1;
    }
    score.0 = count;
}

#[test]
fn test_query_single() {
    let mut world = World::new();
    world.insert_resource(Score(0));

    let e1 = world.spawn_empty();
    let e2 = world.spawn_empty();
    let e3 = world.spawn_empty();
    world.add_component(e1, Position { x: 0.0, y: 0.0 });
    world.add_component(e2, Position { x: 1.0, y: 1.0 });
    world.add_component(e3, Position { x: 2.0, y: 2.0 });

    let mut state = count_positions_init(&mut world);
    count_positions_run(&mut world, &mut state);

    assert_eq!(world.get_resource::<Score>().unwrap().0, 3);
}

// --- Test: Query with pair + mutation ---

#[system]
fn apply_velocity(mut query: Query<(&Position, &mut Velocity)>) {
    for (_entity, (pos, vel)) in &mut query {
        vel.dx += pos.x * 0.1;
        vel.dy += pos.y * 0.1;
    }
}

#[test]
fn test_query_pair_mutation() {
    let mut world = World::new();

    let e = world.spawn_empty();
    world.add_component(e, Position { x: 10.0, y: 20.0 });
    world.add_component(e, Velocity { dx: 0.0, dy: 0.0 });

    let mut state = apply_velocity_init(&mut world);
    apply_velocity_run(&mut world, &mut state);

    let vel = world.get_component::<Velocity>(e).unwrap();
    assert!((vel.dx - 1.0).abs() < f32::EPSILON);
    assert!((vel.dy - 2.0).abs() < f32::EPSILON);
}

// --- Test: Query + ResMut simultaneously (the key use case!) ---

struct SceneData(Vec<String>);

#[system]
fn collect_into_scene(query: Query<&Position>, mut scene: ResMut<SceneData>) {
    scene.0.clear();
    for (_entity, pos) in &query {
        scene.0.push(format!("({}, {})", pos.x, pos.y));
    }
}

#[test]
fn test_query_plus_resmut_simultaneous() {
    let mut world = World::new();
    world.insert_resource(SceneData(Vec::new()));

    let e1 = world.spawn_empty();
    let e2 = world.spawn_empty();
    world.add_component(e1, Position { x: 1.0, y: 2.0 });
    world.add_component(e2, Position { x: 3.0, y: 4.0 });

    let mut state = collect_into_scene_init(&mut world);
    collect_into_scene_run(&mut world, &mut state);

    let scene = world.get_resource::<SceneData>().unwrap();
    assert_eq!(scene.0.len(), 2);
}

// --- Test: SystemAccess is correctly generated ---

#[test]
fn test_access_declarations() {
    let access = increment_score_access();
    // ResMut<Score> writes Score, Res<DeltaTime> reads DeltaTime
    // We can't inspect the BTreeSet directly, but we can check conflict detection
    let other_write_score = {
        let mut a = skesis::SystemAccess::default();
        a.writes_resource::<Score>();
        a
    };
    assert!(access.conflicts_with(&other_write_score));

    let other_read_dt = {
        let mut a = skesis::SystemAccess::default();
        a.reads_resource::<DeltaTime>();
        a
    };
    assert!(!access.conflicts_with(&other_read_dt)); // read-read doesn't conflict
}
