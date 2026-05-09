use pyo3::prelude::*;

const MODE_STANDARD: u8 = 0;
const MODE_ALICE: u8 = 1;
const MODE_LITTLE_ENDIAN: u8 = 2;

// ════════════════════ Operations ════════════════════

#[inline(always)]
fn op_add(a: i32, b: i32) -> Option<i32> {
    Some(a + b)
}
#[inline(always)]
fn op_sub(a: i32, b: i32) -> Option<i32> {
    if a >= b { Some(a - b) } else { None }
}
#[inline(always)]
fn op_rsub(a: i32, b: i32) -> Option<i32> {
    if b >= a { Some(b - a) } else { None }
}
#[inline(always)]
fn op_absdiff(a: i32, b: i32) -> Option<i32> {
    Some((a - b).abs())
}
#[inline(always)]
fn op_mul(a: i32, b: i32) -> Option<i32> {
    Some(a * b)
}
#[inline(always)]
fn op_gcd(a: i32, b: i32) -> Option<i32> {
    Some(gcd(a, b))
}
#[inline(always)]
fn op_lcm(a: i32, b: i32) -> Option<i32> {
    if a == 0 || b == 0 {
        Some(0)
    } else {
        Some(a * b / gcd(a, b))
    }
}
#[inline(always)]
fn op_fdiv(a: i32, b: i32) -> Option<i32> {
    if b != 0 { Some(a / b) } else { None }
}
#[inline(always)]
fn op_rdiv(a: i32, b: i32) -> Option<i32> {
    if a != 0 { Some(b / a) } else { None }
}
#[inline(always)]
fn op_mod(a: i32, b: i32) -> Option<i32> {
    if b != 0 { Some(a % b) } else { None }
}
#[inline(always)]
fn op_rmod(a: i32, b: i32) -> Option<i32> {
    if a != 0 { Some(b % a) } else { None }
}
#[inline(always)]
fn op_min(a: i32, b: i32) -> Option<i32> {
    Some(if a < b { a } else { b })
}
#[inline(always)]
fn op_max(a: i32, b: i32) -> Option<i32> {
    Some(if a > b { a } else { b })
}
#[inline(always)]
fn op_xor(a: i32, b: i32) -> Option<i32> {
    Some(a ^ b)
}
#[inline(always)]
fn op_band(a: i32, b: i32) -> Option<i32> {
    Some(a & b)
}
#[inline(always)]
fn op_bor(a: i32, b: i32) -> Option<i32> {
    Some(a | b)
}

#[inline(always)]
fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.abs()
}

#[inline(always)]
fn apply(op_idx: u8, a: i32, b: i32) -> Option<i32> {
    match op_idx {
        0 => op_add(a, b),
        1 => op_sub(a, b),
        2 => op_rsub(a, b),
        3 => op_absdiff(a, b),
        4 => op_absdiff(a, b), // neg_absdiff same math, sign handled outside
        5 => op_mul(a, b),
        6 => op_gcd(a, b),
        7 => op_lcm(a, b),
        8 => op_fdiv(a, b),
        9 => op_rdiv(a, b),
        10 => op_mod(a, b),
        11 => op_rmod(a, b),
        12 => op_min(a, b),
        13 => op_max(a, b),
        14 => nz(op_add(a, b)? - 1),
        15 => nz(op_add(a, b)? + 1),
        16 => nz(op_mul(a, b)? - 1),
        17 => nz(op_mul(a, b)? + 1),
        18 => nz(op_absdiff(a, b)? - 1),
        19 => nz(op_absdiff(a, b)? + 1),
        20 => nz(op_sub(a, b)? - 1),
        21 => nz(op_sub(a, b)? + 1),
        22 => nz(op_rsub(a, b)? - 1),
        23 => nz(op_rsub(a, b)? + 1),
        24 => nz(op_add(a, b)? - 2),
        25 => nz(op_add(a, b)? + 2),
        26 => nz(op_mul(a, b)? - 2),
        27 => nz(op_mul(a, b)? + 2),
        28 => nz(op_absdiff(a, b)? - 2),
        29 => nz(op_absdiff(a, b)? + 2),
        30 => op_mul(a, b).map(|v| v / 2),
        31 => op_mul(a, b).map(|v| v * 2),
        32 => op_sub(a, b).map(|v| v * v),
        33 => op_add(a, b).map(|v| v * v),
        34 => op_mul(a, b).map(|v| v + a),
        35 => op_mul(a, b).map(|v| v + b),
        36 => nz(op_mul(a, b)? - a),
        37 => nz(op_mul(a, b)? - b),
        38 => Some(a * a + b),
        39 => Some(a + b * b),
        40 => op_xor(a, b),
        41 => op_band(a, b),
        42 => op_bor(a, b),
        // Signed subtractions: return signed value; caller compares sign vs has_sign.
        43 => Some(a - b),
        44 => Some(b - a),
        _ => None,
    }
}

#[inline(always)]
fn is_signed_op(idx: u8) -> bool {
    idx == 43 || idx == 44
}

#[inline(always)]
fn is_concat_op(idx: u8) -> bool {
    idx == 45 || idx == 46
}

#[inline(always)]
fn nz(v: i32) -> Option<i32> {
    if v >= 0 { Some(v) } else { None }
}

// ════════════════════ Data structures ════════════════════

struct EqData {
    l0: u8,
    l1: u8,
    r0: u8,
    r1: u8,
    res_len: u8,
    has_sign: bool,
    res_syms: Vec<u8>,
}

struct OpGroup {
    candidates: Vec<u8>,
    eqs: Vec<EqData>,
}

// ════════════════════ Search ════════════════════

#[inline(always)]
fn pow_i32(base: i32, exp: usize) -> i32 {
    let mut out = 1i32;
    for _ in 0..exp {
        out *= base;
    }
    out
}

#[inline(always)]
fn eq_is_assigned(eq: &EqData, assigned: &[bool]) -> bool {
    if !assigned[eq.l0 as usize]
        || !assigned[eq.l1 as usize]
        || !assigned[eq.r0 as usize]
        || !assigned[eq.r1 as usize]
    {
        return false;
    }
    for &ri in &eq.res_syms {
        if !assigned[ri as usize] {
            return false;
        }
    }
    true
}

#[inline(always)]
fn candidate_matches_eq(cand: u8, eq: &EqData, perm: &[u8], mode: u8, base: i32) -> bool {
    if is_concat_op(cand) {
        if eq.has_sign || eq.res_syms.len() != 4 {
            return false;
        }
        if cand == 45 {
            return eq.res_syms[0] == eq.l0
                && eq.res_syms[1] == eq.l1
                && eq.res_syms[2] == eq.r0
                && eq.res_syms[3] == eq.r1;
        }
        return eq.res_syms[0] == eq.r0
            && eq.res_syms[1] == eq.r1
            && eq.res_syms[2] == eq.l0
            && eq.res_syms[3] == eq.l1;
    }
    let signed_op = is_signed_op(cand);
    let l = perm[eq.l0 as usize] as i32 * base + perm[eq.l1 as usize] as i32;
    let r = perm[eq.r0 as usize] as i32 * base + perm[eq.r1 as usize] as i32;
    let (mut lv, mut rv) = (l, r);
    if mode == MODE_ALICE || mode == MODE_LITTLE_ENDIAN {
        lv = (lv % base) * base + lv / base;
        rv = (rv % base) * base + rv / base;
    }
    let raw = match apply(cand, lv, rv) {
        Some(x) => x,
        None => return false,
    };
    let mut v = if signed_op {
        // Sign of raw result must match whether rhs has a prefix.
        if (raw < 0) != eq.has_sign {
            return false;
        }
        raw.abs()
    } else {
        if raw < 0 {
            return false;
        }
        raw
    };
    let rl = eq.res_len as usize;
    // len(base-N digits of v) > rl  <=>  v >= base^rl
    if v >= pow_i32(base, rl) {
        return false;
    }
    if mode == MODE_ALICE || mode == MODE_LITTLE_ENDIAN {
        // equivalent to: reverse base-N digits after zfill(rl)
        let mut rev = 0i32;
        let mut x = v;
        for _ in 0..rl {
            rev = rev * base + x % base;
            x /= base;
        }
        v = rev;
    }
    // Parse expected value from symbols as base-N digits.
    let mut exp = 0i32;
    for &ri in &eq.res_syms {
        exp = exp * base + perm[ri as usize] as i32;
    }
    v == exp
}

#[inline(always)]
fn group_has_candidate_for_assigned_eqs(
    group: &OpGroup,
    perm: &[u8],
    mode: u8,
    base: i32,
    assigned: &[bool],
) -> bool {
    'cand: for &cand in &group.candidates {
        for eq in &group.eqs {
            if eq_is_assigned(eq, assigned) && !candidate_matches_eq(cand, eq, perm, mode, base) {
                continue 'cand;
            }
        }
        return true;
    }
    false
}

fn search(
    groups: &[OpGroup],
    n: usize,
    mode: u8,
    radix: usize,
    digit_count: usize,
) -> Option<(Vec<u8>, Vec<Vec<u8>>)> {
    let mut perm = vec![0u8; n];
    let mut used = vec![false; digit_count];
    let mut assigned = vec![false; n];
    let mut ops_valid = vec![Vec::new(); groups.len()];
    let var_order = variable_order(groups, n);
    let base = radix as i32;

    fn dfs(
        depth: usize,
        n: usize,
        perm: &mut [u8],
        used: &mut [bool],
        assigned: &mut [bool],
        var_order: &[usize],
        groups: &[OpGroup],
        mode: u8,
        base: i32,
        ops_valid: &mut [Vec<u8>],
    ) -> bool {
        if depth == n {
            // Verify all groups
            for (gi, g) in groups.iter().enumerate() {
                let mut valid = Vec::new();
                'cand: for &cand in &g.candidates {
                    for eq in &g.eqs {
                        if !candidate_matches_eq(cand, eq, perm, mode, base) {
                            continue 'cand;
                        }
                    }
                    valid.push(cand);
                }
                if valid.is_empty() {
                    return false;
                }
                ops_valid[gi] = valid;
            }
            return true;
        }

        let sym_idx = var_order[depth];
        for d in 0..used.len() as u8 {
            if used[d as usize] {
                continue;
            }
            used[d as usize] = true;
            assigned[sym_idx] = true;
            perm[sym_idx] = d;
            let mut still_possible = true;
            for g in groups {
                if !group_has_candidate_for_assigned_eqs(g, perm, mode, base, assigned) {
                    still_possible = false;
                    break;
                }
            }
            if !still_possible {
                assigned[sym_idx] = false;
                used[d as usize] = false;
                continue;
            }
            if dfs(
                depth + 1,
                n,
                perm,
                used,
                assigned,
                var_order,
                groups,
                mode,
                base,
                ops_valid,
            ) {
                return true;
            }
            assigned[sym_idx] = false;
            used[d as usize] = false;
        }
        false
    }

    if dfs(
        0,
        n,
        &mut perm,
        &mut used,
        &mut assigned,
        &var_order,
        groups,
        mode,
        base,
        &mut ops_valid,
    ) {
        Some((perm, ops_valid))
    } else {
        None
    }
}

fn variable_order(groups: &[OpGroup], n: usize) -> Vec<usize> {
    let mut score = vec![0usize; n];
    for g in groups {
        let group_weight = 1 + g.candidates.len();
        for eq in &g.eqs {
            let syms = [eq.l0, eq.l1, eq.r0, eq.r1];
            for &s in &syms {
                score[s as usize] += group_weight;
            }
            for &s in &eq.res_syms {
                score[s as usize] += group_weight;
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(score[i]), i));
    order
}

// ════════════════════ Instrumented search ════════════════════

#[derive(Default, Clone)]
struct SearchStats {
    dfs_calls: usize,
    pruned: usize,           // digit filtered by still_possible (constraint propagation)
    recursed_into: usize,    // digit passed still_possible AND we entered recursion
    recursion_failed: usize, // recursion returned false (== backtrack count, including via leaf failure)
    leaf_failed: usize,      // depth==n verification failed
    max_branch_attempted: usize, // max children we attempted to recurse into at any single depth
}

fn search_with_stats(
    groups: &[OpGroup],
    n: usize,
    mode: u8,
    radix: usize,
    digit_count: usize,
) -> (Option<(Vec<u8>, Vec<Vec<u8>>)>, SearchStats) {
    let mut perm = vec![0u8; n];
    let mut used = vec![false; digit_count];
    let mut assigned = vec![false; n];
    let mut ops_valid = vec![Vec::new(); groups.len()];
    let var_order = variable_order(groups, n);
    let base = radix as i32;
    let mut stats = SearchStats::default();

    fn dfs_t(
        depth: usize,
        n: usize,
        perm: &mut [u8],
        used: &mut [bool],
        assigned: &mut [bool],
        var_order: &[usize],
        groups: &[OpGroup],
        mode: u8,
        base: i32,
        ops_valid: &mut [Vec<u8>],
        stats: &mut SearchStats,
    ) -> bool {
        stats.dfs_calls += 1;
        if depth == n {
            for (gi, g) in groups.iter().enumerate() {
                let mut valid = Vec::new();
                'cand: for &cand in &g.candidates {
                    for eq in &g.eqs {
                        if !candidate_matches_eq(cand, eq, perm, mode, base) {
                            continue 'cand;
                        }
                    }
                    valid.push(cand);
                }
                if valid.is_empty() {
                    stats.leaf_failed += 1;
                    return false;
                }
                ops_valid[gi] = valid;
            }
            return true;
        }

        let sym_idx = var_order[depth];
        let mut attempted_here: usize = 0;
        for d in 0..used.len() as u8 {
            if used[d as usize] {
                continue;
            }
            used[d as usize] = true;
            assigned[sym_idx] = true;
            perm[sym_idx] = d;
            let mut still_possible = true;
            for g in groups {
                if !group_has_candidate_for_assigned_eqs(g, perm, mode, base, assigned) {
                    still_possible = false;
                    break;
                }
            }
            if !still_possible {
                stats.pruned += 1;
                assigned[sym_idx] = false;
                used[d as usize] = false;
                continue;
            }
            attempted_here += 1;
            stats.recursed_into += 1;
            if dfs_t(
                depth + 1,
                n,
                perm,
                used,
                assigned,
                var_order,
                groups,
                mode,
                base,
                ops_valid,
                stats,
            ) {
                if attempted_here > stats.max_branch_attempted {
                    stats.max_branch_attempted = attempted_here;
                }
                return true;
            }
            stats.recursion_failed += 1;
            assigned[sym_idx] = false;
            used[d as usize] = false;
        }
        false
    }

    let ok = dfs_t(
        0,
        n,
        &mut perm,
        &mut used,
        &mut assigned,
        &var_order,
        groups,
        mode,
        base,
        &mut ops_valid,
        &mut stats,
    );
    if ok {
        (Some((perm, ops_valid)), stats)
    } else {
        (None, stats)
    }
}

// ════════════════════ Python binding ════════════════════

/// arithmetic_search(n, alice, groups, radix=None, digit_count=None) -> (perm, ops_valid) | None
/// groups: list of (candidates, equations)
///   candidates: list of u8 op indices
///   equations: list of [l0, l1, r0, r1, res_len, r0, r1, ...]
#[pyfunction]
fn arithmetic_search(
    n: usize,
    alice: bool,
    groups: Vec<(Vec<u8>, Vec<Vec<u8>>)>,
    radix: Option<usize>,
    digit_count: Option<usize>,
    mode: Option<u8>,
) -> PyResult<Option<(Vec<u8>, Vec<Vec<u8>>)>> {
    let rust_groups: Vec<OpGroup> = groups
        .into_iter()
        .map(|(cands, eqs)| {
            let eqs = eqs
                .into_iter()
                .map(|e| EqData {
                    l0: e[0],
                    l1: e[1],
                    r0: e[2],
                    r1: e[3],
                    res_len: e[4],
                    has_sign: e[5] != 0,
                    res_syms: e[6..].to_vec(),
                })
                .collect();
            OpGroup {
                candidates: cands,
                eqs,
            }
        })
        .collect();

    Ok(search(
        &rust_groups,
        n,
        mode.unwrap_or(if alice { MODE_ALICE } else { MODE_STANDARD }),
        radix.unwrap_or(n),
        digit_count.unwrap_or(n),
    ))
}

/// arithmetic_search_with_stats — same as arithmetic_search but also returns DFS stats.
/// Returns (perm_or_none, stats_dict). On success perm is Some((perm, ops_valid)).
/// stats_dict keys: dfs_calls, pruned, recursed_into, recursion_failed, leaf_failed, max_branch_attempted
#[pyfunction]
fn arithmetic_search_with_stats<'py>(
    py: Python<'py>,
    n: usize,
    alice: bool,
    groups: Vec<(Vec<u8>, Vec<Vec<u8>>)>,
    radix: Option<usize>,
    digit_count: Option<usize>,
    mode: Option<u8>,
) -> PyResult<(Option<(Vec<u8>, Vec<Vec<u8>>)>, Bound<'py, pyo3::types::PyDict>)> {
    let rust_groups: Vec<OpGroup> = groups
        .into_iter()
        .map(|(cands, eqs)| {
            let eqs = eqs
                .into_iter()
                .map(|e| EqData {
                    l0: e[0],
                    l1: e[1],
                    r0: e[2],
                    r1: e[3],
                    res_len: e[4],
                    has_sign: e[5] != 0,
                    res_syms: e[6..].to_vec(),
                })
                .collect();
            OpGroup {
                candidates: cands,
                eqs,
            }
        })
        .collect();

    let (result, stats) = search_with_stats(
        &rust_groups,
        n,
        mode.unwrap_or(if alice { MODE_ALICE } else { MODE_STANDARD }),
        radix.unwrap_or(n),
        digit_count.unwrap_or(n),
    );
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("dfs_calls", stats.dfs_calls)?;
    dict.set_item("pruned", stats.pruned)?;
    dict.set_item("recursed_into", stats.recursed_into)?;
    dict.set_item("recursion_failed", stats.recursion_failed)?;
    dict.set_item("leaf_failed", stats.leaf_failed)?;
    dict.set_item("max_branch_attempted", stats.max_branch_attempted)?;
    Ok((result, dict))
}

#[pymodule]
fn alice_sovler_helper(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(arithmetic_search, m)?)?;
    m.add_function(wrap_pyfunction!(arithmetic_search_with_stats, m)?)?;
    Ok(())
}
