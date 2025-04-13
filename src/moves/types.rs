use crate::tsplib::Solution;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CycleId {
    Cycle1,
    Cycle2,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Move {
    InterRouteExchange {
        v1: usize,
        v2: usize,
    },
    IntraRouteVertexExchange {
        v1: usize,
        v2: usize,
        cycle: CycleId,
    },
    IntraRouteEdgeExchange {
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        cycle: CycleId,
    },
}

#[derive(Debug, Clone)]
pub struct EvaluatedMove {
    pub move_type: Move,
    pub delta: i32,
}

impl Move {
    pub fn apply(&self, solution: &mut Solution) {
        match self {
            Move::InterRouteExchange { v1, v2 } => {
                let pos1_opt = solution.find_node(*v1);
                let pos2_opt = solution.find_node(*v2);

                if let (Some((CycleId::Cycle1, pos1)), Some((CycleId::Cycle2, pos2))) =
                    (pos1_opt, pos2_opt)
                {
                    solution.cycle1[pos1] = *v2;
                    solution.cycle2[pos2] = *v1;
                } else if let (Some((CycleId::Cycle2, pos1)), Some((CycleId::Cycle1, pos2))) =
                    (pos1_opt, pos2_opt)
                {
                    solution.cycle2[pos1] = *v2;
                    solution.cycle1[pos2] = *v1;
                } else {
                    eprintln!(
                        "Warning: InterRouteExchange apply failed. Nodes {} or {} not found in expected cycles.",
                        v1, v2
                    );
                }
            }
            Move::IntraRouteVertexExchange { v1, v2, cycle } => {
                if let (Some((c1, pos1)), Some((c2, pos2))) =
                    (solution.find_node(*v1), solution.find_node(*v2))
                {
                    if c1 == *cycle && c2 == *cycle {
                        let cycle_vec = solution.get_cycle_mut(*cycle);
                        cycle_vec.swap(pos1, pos2);
                    } else {
                        eprintln!(
                            "Warning: IntraRouteVertexExchange apply failed. Nodes {} or {} not in cycle {:?}.",
                            v1, v2, cycle
                        );
                    }
                } else {
                    eprintln!(
                        "Warning: IntraRouteVertexExchange apply failed. Nodes {} or {} not found.",
                        v1, v2
                    );
                }
            }
            Move::IntraRouteEdgeExchange {
                a,
                b,
                c,
                d: _,
                cycle,
            } => {
                if let (Some((cb, pos_b)), Some((cc, pos_c))) =
                    (solution.find_node(*b), solution.find_node(*c))
                {
                    if cb == *cycle && cc == *cycle {
                        let cycle_vec = solution.get_cycle_mut(*cycle);
                        let n = cycle_vec.len();
                        if n < 2 {
                            return;
                        }

                        let mut start = pos_b;
                        let mut end = pos_c;

                        if start > end {
                            let mut temp_slice = Vec::with_capacity(n);
                            temp_slice.extend_from_slice(&cycle_vec[start..]);
                            temp_slice.extend_from_slice(&cycle_vec[..=end]);
                            temp_slice.reverse();
                            let mut temp_iter = temp_slice.into_iter();
                            for i in start..n {
                                cycle_vec[i] = temp_iter.next().unwrap();
                            }
                            for i in 0..=end {
                                cycle_vec[i] = temp_iter.next().unwrap();
                            }
                        } else {
                            cycle_vec[start..=end].reverse();
                        }
                    } else {
                        eprintln!(
                            "Warning: IntraRouteEdgeExchange apply failed. Nodes {} or {} not in cycle {:?}.",
                            b, c, cycle
                        );
                    }
                } else {
                    eprintln!(
                        "Warning: IntraRouteEdgeExchange apply failed. Nodes {} or {} not found.",
                        b, c
                    );
                }
            }
        }
    }
}
