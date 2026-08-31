cfg_select! {
    feature = "shuttle" => { use shuttle::thread; }
    _ => { use std::thread; }
}

#[test]
fn race_insert() {
    check_dfs(|| {
        let map = arctic::concurrent::Map::<u64, u64, arctic::concurrent::smr::NoOp>::new();

        thread::scope(|scope| {
            let a = scope.spawn(|| {
                map.insert(5, 3)
                    .map(|value| *value)
                    .map_err(|(value, _)| *value)
            });
            let b = map
                .insert(5, 1)
                .map(|value| *value)
                .map_err(|(value, _)| *value);

            let a = a.join().unwrap();
            match (a, b) {
                (Ok(3), Err(3)) | (Err(1), Ok(1)) => (),
                _ => panic!("Impossible outcome: a={:?}, b={:x?}", a, b),
            }
        });
    });
}

/// A recursive remove trims its key reader while collapsing empty nodes; the
/// trimmed integer reader used to keep the removed key's trailing bytes in its
/// buffer, so a remove retry racing a neighbor-insert split plus a reinsert of
/// the same key could over-match a value edge and hit
/// `unreachable!("Prefix condition")` in `Cursor::traverse_node`.
#[test]
fn race_remove_split_reinsert() {
    check_dfs(|| {
        let map = arctic::concurrent::Map::<u128, u64, arctic::concurrent::smr::NoOp>::new();

        // 16-byte keys so the path contains chained single-child nodes,
        // giving the recursive remove something to collapse.
        const KEY: u128 = 0x0101_0101_0101_0101_0101_0101_0101_0101;
        // Adjacent key: shares every byte except the last, so inserting it
        // splits the deepest edge along `KEY`'s path.
        const NEIGHBOR: u128 = KEY + 1;

        map.insert(KEY, 1).map(|value| *value).unwrap();

        thread::scope(|scope| {
            let remove = scope.spawn(|| {
                map.remove(&KEY);
            });

            let _ = map.insert(NEIGHBOR, 2).map(|value| *value);
            let _ = map.insert(KEY, 3).map(|value| *value);

            remove.join().unwrap();

            assert_eq!(map.get(&NEIGHBOR).as_deref(), Some(&2));
        });
    });
}

fn check_dfs<F>(run: F)
where
    F: Fn() + Send + Sync + 'static,
{
    cfg_select! {
        feature = "shuttle" => { shuttle::check_dfs(run, None); }
        _ => {
            for _ in 0..1000 {
                run();
            }
        }
    }
}
