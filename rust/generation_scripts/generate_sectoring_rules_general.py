#!/usr/bin/env python3
import yaml
import itertools
from pprint import pprint
import os
pjoin = os.path.join


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


root_path = os.path.dirname(os.path.realpath(__file__))

template = """# The cut ordering is as follows:
# | ID   (cut edges   ) : E-surf ID
# | #0   (-0|+1       ) : #21
# | #1   (-2|+3       ) : #10
# | #2   (-0|+4|+6    ) : #19
# | #3   (+1|-4|-5    ) : #22
# | #4   (+6|-7|-2    ) : #17
# | #5   (+3|+7|-5    ) : #11
# | #6   (-0|+4|+7|+3 ) : #15
# | #7   (+1|-4|-7|-2 ) : #23
# | #8   (-5|+6       ) : #18
# And the loop momenta indices conventions are k=0 (left), m=1 (middle), l=2 (right)
# If a CT belongs to a sector and cut not matching any rule below, the default behaviour will be applied
# Convention for the signature of each cut:
# -2: absent from cFF term OR inactive
# -1: absent from cFF term
#  0: present in cFF term but inactive cut (selected away by observable)
#  1: present in cFF term and active cut (selected by the observable)
%(rules)s
"""
CUT_ABSENT_OR_INACTIVE = -2
CUT_ABSENT = -1
CUT_INACTIVE = 0
CUT_ACTIVE = 1

CUT_01 = 0
CUT_23 = 1
CUT_046 = 2
CUT_145 = 3
CUT_672 = 4
CUT_375 = 5
CUT_0473 = 6
CUT_1472 = 7
CUT_56 = 8

E_SURF_MAP = [21, 10, 19, 22, 17, 11, 15, 23, 18]

K = 0
L = 1
M = 2


def intersection_info(edges, specified_info=None):
    default = {
        'loop_indices_solved': [],
        'edges': edges
    }
    if specified_info is not None:
        default.update(specified_info)
    return default


THRESHOLD_INTERSECTION_INFO = {
    CUT_01: {
        CUT_01: intersection_info([0, 1]),
        CUT_23: intersection_info([2, 3], {'loop_indices_solved': [[L, M], [L]]}),
        CUT_046: intersection_info([0, 4, 6]),
        CUT_145: intersection_info([1, 4, 5]),
        CUT_672: intersection_info([6, 7, 2], {'loop_indices_solved': [[L, M], [L]]}),
        CUT_375: intersection_info([3, 7, 5], {'loop_indices_solved': [[L, M], [L]]}),
        CUT_0473: intersection_info([0, 4, 7, 3]),
        CUT_1472: intersection_info([1, 4, 7, 2]),
        CUT_56: intersection_info([5, 6], {'loop_indices_solved': [[L, M], [M]]}),
    },
    CUT_23: {
        CUT_01: intersection_info([0, 1], {'loop_indices_solved': [[K, M], [K]]}),
        CUT_23: intersection_info([2, 3]),
        CUT_046: intersection_info([0, 4, 6], {'loop_indices_solved': [[K, M], [K]]}),
        CUT_145: intersection_info([1, 4, 5], {'loop_indices_solved': [[K, M], [K]]}),
        CUT_672: intersection_info([6, 7, 2]),
        CUT_375: intersection_info([3, 7, 5]),
        CUT_0473: intersection_info([0, 4, 7, 3]),
        CUT_1472: intersection_info([1, 4, 7, 2]),
        CUT_56: intersection_info([5, 6], {'loop_indices_solved': [[K, M], [M]]}),
    },
    CUT_046: {
        CUT_01: intersection_info([0, 1]),
        CUT_23: intersection_info([2, 3], {'loop_indices_solved': [[L]]}),
        CUT_046: intersection_info([0, 4, 6]),
        CUT_145: intersection_info([1, 4, 5]),
        CUT_672: intersection_info([6, 7, 2], {'loop_indices_solved': [[L]]}),
        CUT_375: intersection_info([3, 7, 5], {'loop_indices_solved': [[L]]}),
        CUT_0473: intersection_info([0, 4, 7, 3]),
        CUT_1472: intersection_info([1, 4, 7, 2]),
        CUT_56: intersection_info([5, 6]),
    },
    CUT_145: {
        CUT_01: intersection_info([0, 1]),
        CUT_23: intersection_info([2, 3], {'loop_indices_solved': [[L]]}),
        CUT_046: intersection_info([0, 4, 6]),
        CUT_145: intersection_info([1, 4, 5]),
        CUT_672: intersection_info([6, 7, 2], {'loop_indices_solved': [[L]]}),
        CUT_375: intersection_info([3, 7, 5], {'loop_indices_solved': [[L]]}),
        CUT_0473: intersection_info([0, 4, 7, 3]),
        CUT_1472: intersection_info([1, 4, 7, 2]),
        CUT_56: intersection_info([5, 6]),
    },
    CUT_672: {
        CUT_01: intersection_info([0, 1], {'loop_indices_solved': [[K]]}),
        CUT_23: intersection_info([2, 3]),
        CUT_046: intersection_info([0, 4, 6], {'loop_indices_solved': [[K]]}),
        CUT_145: intersection_info([1, 4, 5], {'loop_indices_solved': [[K]]}),
        CUT_672: intersection_info([6, 7, 2]),
        CUT_375: intersection_info([3, 7, 5]),
        CUT_0473: intersection_info([0, 4, 7, 3]),
        CUT_1472: intersection_info([1, 4, 7, 2]),
        CUT_56: intersection_info([5, 6]),
    },
    CUT_375: {
        CUT_01: intersection_info([0, 1], {'loop_indices_solved': [[K]]}),
        CUT_23: intersection_info([2, 3]),
        CUT_046: intersection_info([0, 4, 6], {'loop_indices_solved': [[K]]}),
        CUT_145: intersection_info([1, 4, 5], {'loop_indices_solved': [[K]]}),
        CUT_672: intersection_info([6, 7, 2]),
        CUT_375: intersection_info([3, 7, 5]),
        CUT_0473: intersection_info([0, 4, 7, 3]),
        CUT_1472: intersection_info([1, 4, 7, 2]),
        CUT_56: intersection_info([5, 6]),
    },
    CUT_0473: {
        CUT_01: intersection_info([0, 1],),
        CUT_23: intersection_info([2, 3]),
        CUT_046: intersection_info([0, 4, 6]),
        CUT_145: intersection_info([1, 4, 5]),
        CUT_672: intersection_info([6, 7, 2]),
        CUT_375: intersection_info([3, 7, 5]),
        CUT_0473: intersection_info([0, 4, 7, 3]),
        CUT_1472: intersection_info([1, 4, 7, 2]),
        CUT_56: intersection_info([5, 6]),
    },
    CUT_1472: {
        CUT_01: intersection_info([0, 1]),
        CUT_23: intersection_info([2, 3]),
        CUT_046: intersection_info([0, 4, 6]),
        CUT_145: intersection_info([1, 4, 5]),
        CUT_672: intersection_info([6, 7, 2]),
        CUT_375: intersection_info([3, 7, 5]),
        CUT_0473: intersection_info([0, 4, 7, 3]),
        CUT_1472: intersection_info([0, 4, 7, 3]),
        CUT_56: intersection_info([5, 6]),
    },
    CUT_56: {
        CUT_01: intersection_info([0, 1], {'loop_indices_solved': [[K]]}),
        CUT_23: intersection_info([2, 3], {'loop_indices_solved': [[L]]}),
        CUT_046: intersection_info([0, 4, 6]),
        CUT_145: intersection_info([1, 4, 5]),
        CUT_672: intersection_info([6, 7, 2]),
        CUT_375: intersection_info([3, 7, 5]),
        CUT_0473: intersection_info([0, 4, 7, 3]),
        CUT_1472: intersection_info([0, 4, 7, 3]),
        CUT_56: intersection_info([5, 6]),
    },
}

CUT_DEPENDENCIES = {
    K: [CUT_01, CUT_145, CUT_046],
    L: [CUT_23, CUT_375, CUT_672],
    M: [CUT_56, CUT_145, CUT_046, CUT_375, CUT_672]
}
CUT_LOOP_MOMENTA_INDICES = {
    CUT_01: [K],
    CUT_23: [L],
    CUT_046: [K, M],
    CUT_145: [K, M],
    CUT_672: [L, M],
    CUT_375: [L, M],
    CUT_0473: [K, L, M],
    CUT_1472: [K, L, M],
    CUT_56: [M],
}

NO_MC_FACTOR = {'e_surf_ids_prod_in_num': [],
                'e_surf_ids_prods_to_sum_in_denom': []}


def does_e_surf_id_exist_in_cff_factor(e_surf_id, factor):
    if e_surf_id in factor['denominator']:
        return True
    for f in factor['factors']:
        if does_e_surf_id_exist_in_cff_factor(e_surf_id, f):
            return True
    return False


COMBINE_ABSENT_AND_INACTIVE = False


def generate_file(filename):

    TriBoxTri_yaml = None
    with open(pjoin(root_path, '../data/TriBoxTri.yaml'), 'r') as f:
        TriBoxTri_yaml = yaml.load(f, Loader=yaml.FullLoader)
    cff_expression = TriBoxTri_yaml['supergraph_cff']['cff_expression']
    possible_signatures_for_always_active_cut = []
    for cff_term in cff_expression['terms']:
        new_sig = [
            (CUT_ACTIVE if any(does_e_surf_id_exist_in_cff_factor(e_surf_id, f) for f in cff_term['factors']) else CUT_ABSENT) for e_surf_id in E_SURF_MAP
        ]
        if new_sig not in possible_signatures_for_always_active_cut:
            possible_signatures_for_always_active_cut.append(new_sig)
    print("When assuming all cuts active, there are %d possible sectors." %
          len(possible_signatures_for_always_active_cut))
    possible_signatures = []
    for p in possible_signatures_for_always_active_cut:
        possible_signatures.extend(itertools.product(
            *[[CUT_ABSENT,] if s == CUT_ABSENT else [CUT_INACTIVE, CUT_ACTIVE] for s in p]))

    if COMBINE_ABSENT_AND_INACTIVE:
        new_sigs = []
        for p in possible_signatures:
            new_sig = [
                (CUT_ACTIVE if s == CUT_ACTIVE else CUT_ABSENT_OR_INACTIVE) for s in p]
            if new_sig not in new_sigs:
                new_sigs.append(new_sig)
        possible_signatures = new_sigs

    if COMBINE_ABSENT_AND_INACTIVE:
        print("When assuming all possible combinations of active, inactive OR absent cuts, there are %d possible sectors" %
              len(possible_signatures))
    else:
        print("When assuming all possible combinations of active, inactive AND absent cuts, there are %d possible sectors" %
              len(possible_signatures))

    rules = []
    for i_rule, sig in enumerate(possible_signatures):
        print("Generating rule for sector [%s] %-3d/%s" %
              (','.join(('+%d' % s if s >= 0 else '%d' % s) for s in sig), i_rule+1, len(possible_signatures)), end='\r')

        # sector_rule = {
        #     'sector_signature': list(sig),
        #     'rules_for_cut': [ {
        #         'cut_id': 1,
        #         'rules_for_ct': [
        #             {
        #                 'surf_id_subtracted': 18,
        #                 'loop_indices_this_ct_is_solved_in': [0, 2],
        #                 'enabled': False,
        #                 'mc_factor': {
        #                     'e_surf_ids_prod_in_num': [[10, 11]],
        #                     'e_surf_ids_prods_to_sum_in_denom': [[[10, 11]], [[10, 18]]]
        #                 }
        #             }
        #         ]
        #     }, ]
        # }
        sector_rule = {'sector_signature': list(sig), 'rules_for_cut': []}

        for cut_to_add in [i_s for i_s, s in enumerate(sig) if s == CUT_ACTIVE]:
            rules_for_ct = []
            for intersecting_cut, intersecting_cut_info in THRESHOLD_INTERSECTION_INFO[cut_to_add].items():
                if len(intersecting_cut_info['loop_indices_solved']) == 0:
                    continue
                max_loop_indices = list(sorted(max(
                    intersecting_cut_info['loop_indices_solved'], key=lambda x: len(x))))

                if len(intersecting_cut_info['loop_indices_solved']) == 2 and len(max_loop_indices) == 2:
                    orthogonal_space = [i for i in max_loop_indices if not all(
                        i in lis for lis in intersecting_cut_info['loop_indices_solved'])][0]
                    common_space = [i for i in max_loop_indices if all(
                        i in lis for lis in intersecting_cut_info['loop_indices_solved'])][0]
                    # Include both spaces for solving but PFed using the E-surface of all active cuts intersecting the subspace chosen
                    active_thresholds_in_orthogonal_space = [
                        c for c in CUT_DEPENDENCIES[orthogonal_space] if c != intersecting_cut and CUT_LOOP_MOMENTA_INDICES[c] != max_loop_indices and sig[c] == CUT_ACTIVE]
                    inactive_thresholds_in_orthogonal_space = [
                        c for c in CUT_DEPENDENCIES[orthogonal_space] if c != intersecting_cut and CUT_LOOP_MOMENTA_INDICES[c] != max_loop_indices and sig[c] == CUT_INACTIVE]
                    # if sig == (0, 1, -1, 0, -1, 1, -1, -1, 1):
                    #     print('')
                    #     print('sig[3]=', sig[3] == CUT_ACTIVE)
                    #     print("active_thresholds_in_orthogonal_space=",
                    #           active_thresholds_in_orthogonal_space)
                    #     print("inactive_thresholds_in_orthogonal_space=",
                    #           inactive_thresholds_in_orthogonal_space)
                    # active_thresholds_in_common_space = [
                    #     c for c in CUT_DEPENDENCIES[common_space] if c!=intersecting_cut and sig[c] == CUT_ACTIVE]
                    # inactive_thresholds_in_common_space = [
                    #     c for c in CUT_DEPENDENCIES[common_space] if c!=intersecting_cut and sig[c] == CUT_INACTIVE]
                    mc_factor_remove_orthogonal_space_inactive_thresholds = [
                        [E_SURF_MAP[thres], E_SURF_MAP[cut_to_add]] for thres in inactive_thresholds_in_orthogonal_space]
                    mc_factor_remove_orthogonal_space_active_thresholds = [
                        [E_SURF_MAP[thres], E_SURF_MAP[cut_to_add]] for thres in active_thresholds_in_orthogonal_space]
                    if len(mc_factor_remove_orthogonal_space_inactive_thresholds) > 0 and len(mc_factor_remove_orthogonal_space_active_thresholds) > 0:
                        mc_factor_denom = [mc_factor_remove_orthogonal_space_inactive_thresholds,
                                           mc_factor_remove_orthogonal_space_active_thresholds]
                    else:
                        mc_factor_denom = []

                for loop_indices in intersecting_cut_info['loop_indices_solved']:
                    # if sig == (0, 1, -1, 0, -1, 1, -1, -1, 1):
                    #     print("cut=", cut_to_add)
                    #     print("loop_indices=", loop_indices)
                    # Anti-observable
                    if sig[intersecting_cut] == CUT_ACTIVE:
                        is_enabled = False
                        mc_factor = NO_MC_FACTOR
                    else:
                        if len(max_loop_indices) == 1:
                            is_enabled = True
                            mc_factor = NO_MC_FACTOR
                        else:
                            # Include both spaces for solving but PFed using the E-surface of all active cuts intersecting the subspace chosen
                            if max_loop_indices == loop_indices:
                                if mc_factor_denom != []:
                                    is_enabled = True
                                    mc_factor = {
                                        'e_surf_ids_prod_in_num': mc_factor_remove_orthogonal_space_active_thresholds,
                                        'e_surf_ids_prods_to_sum_in_denom': mc_factor_denom
                                    }
                                else:
                                    is_enabled = len(
                                        mc_factor_remove_orthogonal_space_active_thresholds) == 0
                                    mc_factor = NO_MC_FACTOR

                            else:
                                if mc_factor_denom != []:
                                    is_enabled = True
                                    mc_factor = {
                                        'e_surf_ids_prod_in_num': mc_factor_remove_orthogonal_space_inactive_thresholds,
                                        'e_surf_ids_prods_to_sum_in_denom': mc_factor_denom
                                    }
                                else:
                                    is_enabled = len(mc_factor_remove_orthogonal_space_inactive_thresholds) == 0 and len(
                                        mc_factor_remove_orthogonal_space_active_thresholds) > 0
                                    mc_factor = NO_MC_FACTOR

                    # if sig == (0, 1, -1, 0, -1, 1, -1, -1, 1):
                    #     print("mc_factor=", mc_factor)

                    rules_for_ct.append({
                        'surf_id_subtracted': E_SURF_MAP[intersecting_cut],
                        'loop_indices_this_ct_is_solved_in': loop_indices,
                        'enabled': is_enabled,
                        'mc_factor': mc_factor
                    })

            sector_rule['rules_for_cut'].append({
                'cut_id': cut_to_add,
                'rules_for_ct': rules_for_ct
            })

        rules.append(sector_rule)
        continue
    print(' '*63, end='\r')
    # print('')
    # pprint(rules[502])
    print("A total of %d/%d rules have been generated." % (
        len(rules), len(possible_signatures)
    ))
    print("Writing yaml file to '%s' ..." % filename)
    with open(filename, 'w') as f:
        f.write(template % {"rules": yaml.dump(
            rules, default_flow_style=None, Dumper=NoAliasDumper)})
    print("Done")


if __name__ == "__main__":
    generate_file(
        pjoin(root_path, "../data/auto_sectoring_rules_TriBoxTri.yaml"))
