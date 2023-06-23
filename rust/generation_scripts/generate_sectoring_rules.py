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
        CUT_375: intersection_info([3, 7, 5]),
        CUT_0473: intersection_info([0, 4, 7, 3]),
        CUT_1472: intersection_info([1, 4, 7, 2]),
        CUT_56: intersection_info([5, 6]),
    },
    CUT_145: {
        CUT_01: intersection_info([0, 1]),
        CUT_23: intersection_info([2, 3], {'loop_indices_solved': [[L]]}),
        CUT_046: intersection_info([0, 4, 6]),
        CUT_145: intersection_info([1, 4, 5]),
        CUT_672: intersection_info([6, 7, 2]),
        CUT_375: intersection_info([3, 7, 5], {'loop_indices_solved': [[L]]}),
        CUT_0473: intersection_info([0, 4, 7, 3]),
        CUT_1472: intersection_info([1, 4, 7, 2]),
        CUT_56: intersection_info([5, 6]),
    },
    CUT_672: {
        CUT_01: intersection_info([0, 1], {'loop_indices_solved': [[K]]}),
        CUT_23: intersection_info([2, 3]),
        CUT_046: intersection_info([0, 4, 6], {'loop_indices_solved': [[K]]}),
        CUT_145: intersection_info([1, 4, 5]),
        CUT_672: intersection_info([6, 7, 2]),
        CUT_375: intersection_info([3, 7, 5]),
        CUT_0473: intersection_info([0, 4, 7, 3]),
        CUT_1472: intersection_info([1, 4, 7, 2]),
        CUT_56: intersection_info([5, 6]),
    },
    CUT_375: {
        CUT_01: intersection_info([0, 1], {'loop_indices_solved': [[K]]}),
        CUT_23: intersection_info([2, 3]),
        CUT_046: intersection_info([0, 4, 6]),
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

        if sig[CUT_01] == CUT_ACTIVE and (sig[CUT_145] == CUT_ACTIVE or sig[CUT_046] == CUT_ACTIVE) and sig[CUT_56] == CUT_INACTIVE and (sig[CUT_375] == CUT_ACTIVE or sig[CUT_672] == CUT_ACTIVE):
            # TODO handle special "lone56" sector
            continue

        # Create a rule for the broken soft sectors
        broken_soft_sector_info = {}
        if sig[CUT_01] == CUT_ACTIVE and (sig[CUT_145] == CUT_ACTIVE or sig[CUT_046] == CUT_ACTIVE) and sig[CUT_56] == CUT_INACTIVE:
            broken_soft_sector_info = {
                'active_VV_cut': CUT_01,
                'active_RV_cut': CUT_145 if sig[CUT_145] == CUT_ACTIVE else CUT_046,
            }
        if sig[CUT_23] == CUT_ACTIVE and (sig[CUT_375] == CUT_ACTIVE or sig[CUT_672] == CUT_ACTIVE) and sig[CUT_56] == CUT_INACTIVE:
            broken_soft_sector_info = {
                'active_VV_cut': CUT_23,
                'active_RV_cut': CUT_375 if sig[CUT_375] == CUT_ACTIVE else CUT_672,
            }
        if broken_soft_sector_info != {}:
            for cut_to_add in [broken_soft_sector_info['active_VV_cut'], broken_soft_sector_info['active_RV_cut']]+[i_s for i_s, s in enumerate(sig) if s == CUT_ACTIVE if i_s not in [broken_soft_sector_info['active_VV_cut'], broken_soft_sector_info['active_RV_cut']]]:
                rules_for_ct = []
                for intersecting_cut, intersecting_cut_info in THRESHOLD_INTERSECTION_INFO[cut_to_add].items():
                    if len(intersecting_cut_info['loop_indices_solved']) == 0:
                        continue
                    for loop_indices in intersecting_cut_info['loop_indices_solved']:
                        if cut_to_add == broken_soft_sector_info['active_VV_cut']:
                            if intersecting_cut == CUT_56 and len(loop_indices) == 1:
                                is_enabled = False
                                mc_factor = NO_MC_FACTOR
                            elif len(loop_indices) == 2:
                                is_enabled = True
                                mc_factor = {
                                    'e_surf_ids_prod_in_num':
                                        [[E_SURF_MAP[broken_soft_sector_info['active_VV_cut']],
                                            E_SURF_MAP[broken_soft_sector_info['active_RV_cut']]]],
                                        'e_surf_ids_prods_to_sum_in_denom': [
                                            [[E_SURF_MAP[broken_soft_sector_info['active_VV_cut']],
                                                E_SURF_MAP[broken_soft_sector_info['active_RV_cut']]]],
                                            [[E_SURF_MAP[broken_soft_sector_info['active_VV_cut']],
                                                E_SURF_MAP[CUT_56]]]
                                        ]
                                }
                            elif len(loop_indices) == 1 and len(intersecting_cut_info['edges']) == 2:
                                is_enabled = True
                                mc_factor = {
                                    'e_surf_ids_prod_in_num':
                                        [[E_SURF_MAP[broken_soft_sector_info['active_VV_cut']],
                                            E_SURF_MAP[CUT_56]]],
                                        'e_surf_ids_prods_to_sum_in_denom': [
                                            [[E_SURF_MAP[broken_soft_sector_info['active_VV_cut']],
                                                E_SURF_MAP[broken_soft_sector_info['active_RV_cut']]]],
                                            [[E_SURF_MAP[broken_soft_sector_info['active_VV_cut']],
                                                E_SURF_MAP[CUT_56]]]
                                        ]
                                }
                            elif len(loop_indices) == 1 and len(intersecting_cut_info['edges']) > 2:
                                is_enabled = False
                                mc_factor = NO_MC_FACTOR
                        elif cut_to_add == broken_soft_sector_info['active_RV_cut']:
                            if len(intersecting_cut_info['edges']) > 2:
                                is_enabled = True
                                mc_factor = NO_MC_FACTOR
                            else:
                                is_enabled = True
                                mc_factor = {
                                    'e_surf_ids_prod_in_num':
                                        [[E_SURF_MAP[broken_soft_sector_info['active_VV_cut']],
                                            E_SURF_MAP[CUT_56]]],
                                        'e_surf_ids_prods_to_sum_in_denom': [
                                            [[E_SURF_MAP[broken_soft_sector_info['active_VV_cut']],
                                                E_SURF_MAP[broken_soft_sector_info['active_RV_cut']]]],
                                            [[E_SURF_MAP[broken_soft_sector_info['active_VV_cut']],
                                                E_SURF_MAP[CUT_56]]]
                                        ]
                                }
                        elif len(intersecting_cut_info['edges']) == 2:
                            if len(loop_indices) == 1:
                                is_enabled = False
                                mc_factor = NO_MC_FACTOR
                            elif len(loop_indices) > 1:
                                is_enabled = True
                                mc_factor = NO_MC_FACTOR
                        else:
                            # Use default behaviour for other non-two-loop cuts
                            continue

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
