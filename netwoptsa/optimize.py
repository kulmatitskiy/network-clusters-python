from numpy import arange, argmax, transpose as t, matrix, diagonal, outer, unique, repeat
from numpy import random

inf = float("inf")

def argmax_coord(matrix):
    return divmod(argmax(matrix), matrix.shape[1])

def diag(mat):
    return t(matrix(diagonal(matrix(mat))))

def comembership_matrix(groups):
    g = groups.astype(float) + 1
    return matrix(outer(g, 1/g) == 1.0).astype(float)

def switch_groups (groups, D):
    i, j = argmax_coord(D)
    groups[i] = groups[j]
    return groups

def local_steepest_ascent(obj_mat, groups, store_trace=False, trace_apply_fun=None):
    trace_group_counts = list()
    trace_obj_values = list()
    trace_apply_values = list()

    while True:
        # 1. Compute 'Differential Matrix' D as defined in README.MD
        BC = obj_mat * comembership_matrix(groups)
        BCdiag = diag(BC)
        D = BC - BCdiag
        obj_val = sum(BCdiag)

        # store trace data if needed
        if store_trace:
            trace_obj_values.append(obj_value)
            trace_group_counts.append(len(unique(groups)))
            if trace_apply_fun:
                trace_apply_values.append(trace_apply_fun(groups))

        # 2. In any positive differential D[i,j] is found, switch group memberships
        if any(D > 0):
            switch_groups(groups, D)
        else:
            break

    ret = {}
    ret["groups"] = groups
    ret["D"] = D
    ret["obj_val"] = obj_val
    if store_trace:
        ret["trace_obj_values"] = trace_obj_values
        ret["trace_group_counts"] = trace_group_counts
        ret["trace_apply_values"] = trace_apply_values
    return ret


# Sampling methods for random group vectors

def sample_groups_uniform(N, num_groups, prev_groups=None):
    return random.choice(arange(num_groups)+1, N)

def global_steepest_ascent(obj_mat, max_groups, trials=10, reference_groups=None,
                                   details=None, generator=sample_groups_uniform, seed=None):
    if seed:
        random.seed(seed)

    n, _ = obj_mat.shape

    if details:
        # ptm = proc.time() (TODO)
        hits = list()
        wait_times = list()

    best_obj_val = -inf
    best_groups = repeat(1, max_groups)
    reference_obj_val = -inf

    if reference_groups:
        tmp = local_steepest_ascent(obj_mat, reference_groups)
        best_obj_val = tmp["obj_value"]
        best_groups = tmp["groups"]
        reference_obj_val = tmp["obj_value"]

    # Actual search loop begins here
    groups = generator(n, max_groups, None)
    for i in range(trials):
        tmp = local_steepest_ascent(obj_mat, groups)
        if tmp["obj_val"] > best_obj_val:
            best_obj_val = tmp["obj_val"]
            best_groups = tmp["groups"]

            if details:
                wait_times.append((i + 1) if len(hits) == 0 else i - hits[len(hits)-1])
                hits.append(i + 1)

        if i + 1 != trials:
            groups = generator(n, max_groups, group)

    # Done; Prepare return dict
    ret = dict()
    ret["groups"] = best_groups
    ret["obj_val"] = best_obj_val
    if details:
        #ret["time"] = processor time - ptm (TODO)
        if reference_groups:
            ret["reference_obj_val"] = reference_obj_val
        ret["hits"] = hits
        wait_times.append(trials - hits[length(hits) - 1])
        ret["wait_times"] = wait_times
        ret["trials"] = trials
        if seed:
            ret["seed"] = seed
    return ret
