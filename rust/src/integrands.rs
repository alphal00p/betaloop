use crate::box_subtraction::{BoxSubtractionIntegrand, BoxSubtractionSettings};
use crate::h_function_test::{HFunctionTestIntegrand, HFunctionTestSettings};
use crate::loop_induced_triboxtri::{LoopInducedTriBoxTriIntegrand, LoopInducedTriBoxTriSettings};
use crate::triangle_subtraction::{TriangleSubtractionIntegrand, TriangleSubtractionSettings};
use crate::triboxtri::{TriBoxTriIntegrand, TriBoxTriSettings};
use crate::utils::FloatLike;
use crate::{utils, Settings};
use color_eyre::{Help, Report};
use enum_dispatch::enum_dispatch;
use eyre::WrapErr;
use havana::Sample;
use havana::{ContinuousGrid, Grid};
use lorentz_vector::LorentzVector;
use num::Complex;
use serde::Deserialize;
use std::fmt::{Display, Formatter};
use std::fs::File;

#[derive(Debug, Copy, Clone, PartialEq, Deserialize)]
pub enum HardCodedIntegrand {
    #[serde(rename = "unit_surface")]
    UnitSurface,
    #[serde(rename = "unit_volume")]
    UnitVolume,
    #[serde(rename = "loop_induced_TriBoxTri")]
    LoopInducedTriBoxTri,
    #[serde(rename = "triangle_subtraction")]
    TriangleSubtraction,
    #[serde(rename = "box_subtraction")]
    BoxSubtraction,
    #[serde(rename = "h_function_test")]
    HFunctionTest,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(non_snake_case)]
#[serde(tag = "type")]
pub enum HardCodedIntegrandSettings {
    #[serde(rename = "unit_surface")]
    UnitSurface(UnitSurfaceSettings),
    #[serde(rename = "unit_volume")]
    UnitVolume(UnitVolumeSettings),
    #[serde(rename = "loop_induced_TriBoxTri")]
    LoopInducedTriBoxTri(LoopInducedTriBoxTriSettings),
    #[serde(rename = "TriBoxTri")]
    TriBoxTri(TriBoxTriSettings),
    #[serde(rename = "triangle_subtraction")]
    TriangleSubtraction(TriangleSubtractionSettings),
    #[serde(rename = "box_subtraction")]
    BoxSubtraction(BoxSubtractionSettings),
    #[serde(rename = "h_function_test")]
    HFunctionTest(HFunctionTestSettings),
}

impl Display for HardCodedIntegrandSettings {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            HardCodedIntegrandSettings::UnitSurface(_) => write!(f, "unit_surface"),
            HardCodedIntegrandSettings::UnitVolume(_) => write!(f, "unit_volume"),
            HardCodedIntegrandSettings::LoopInducedTriBoxTri(_) => {
                write!(f, "loop_induced_TriBoxTri")
            }
            HardCodedIntegrandSettings::TriBoxTri(_) => {
                write!(f, "TriBoxTri")
            }
            HardCodedIntegrandSettings::TriangleSubtraction(_) => {
                write!(f, "triangle_subtraction")
            }
            HardCodedIntegrandSettings::BoxSubtraction(_) => {
                write!(f, "box_subtraction")
            }
            HardCodedIntegrandSettings::HFunctionTest(_) => {
                write!(f, "h_function_test")
            }
        }
    }
}

impl Default for HardCodedIntegrandSettings {
    fn default() -> HardCodedIntegrandSettings {
        HardCodedIntegrandSettings::UnitSurface(UnitSurfaceSettings { n_3d_momenta: 1 })
    }
}

pub struct ComputationCache<T: FloatLike> {
    pub external_momenta: Vec<LorentzVector<T>>,
}

impl<T: FloatLike> ComputationCache<T> {
    pub fn default() -> ComputationCache<T> {
        ComputationCache {
            external_momenta: vec![],
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Esurface {
    pub edge_ids: Vec<usize>,
    pub shift: Vec<isize>,
    pub id: usize,
}
#[allow(unused)]
impl Esurface {
    pub fn new(edge_ids: Vec<usize>, shift: Vec<isize>, id: usize) -> Esurface {
        Esurface {
            edge_ids,
            shift,
            id,
        }
    }
}

#[derive(Debug)]
pub struct ESurfaceIntegratedCT<T: FloatLike> {
    pub adjusted_sampling_jac: T,
    pub h_function_wgt: T,
    pub e_surf_residue: T,
}

#[derive(Debug)]
pub struct ESurfaceCT<T: FloatLike, ESC: ESurfaceCacheTrait<T>> {
    pub e_surf_id: usize,
    pub ct_basis_signature: Vec<Vec<isize>>,
    pub center_coordinates: Vec<[T; 3]>,
    pub adjusted_sampling_jac: T,
    pub h_function_wgt: T,
    pub e_surf_expanded: T,
    pub loop_momenta_star: Vec<LorentzVector<T>>,
    pub onshell_edges: Vec<LorentzVector<T>>,
    pub e_surface_evals: [Vec<ESC>; 2],
    pub cff_evaluations: [Vec<T>; 2],
    pub solution_type: usize,
    pub ct_level: usize, // either utils::AMPLITUDE_LEVEL_CT or utils::SUPERGRAPH_LEVEL_CT
    pub integrated_ct: Option<ESurfaceIntegratedCT<T>>,
}

pub trait ESurfaceCacheTrait<T: FloatLike> {
    fn does_exist(&self) -> bool;
    fn eval(&self, k: &Vec<LorentzVector<T>>) -> T;
    fn cached_eval(&self) -> T;
    fn norm(&self, k: &Vec<LorentzVector<T>>) -> LorentzVector<T>;
    fn compute_t_scaling(&self, k: &Vec<LorentzVector<T>>) -> [T; 2];
}

#[derive(Debug, Clone)]
pub struct GenericESurfaceCache<T: FloatLike> {
    pub sigs: Vec<Vec<T>>,
    pub ps: Vec<[T; 3]>,
    pub ms: Vec<T>,
    pub e_shift: T,
    pub exists: bool,
    pub eval: T,
    pub t_scaling: [T; 2],
}

#[derive(Debug, Clone)]
pub struct OneLoopESurfaceCache<T: FloatLike> {
    pub p1: [T; 3],
    pub p2: [T; 3],
    pub m1_sq: T,
    pub m2_sq: T,
    pub e_shift: T,
    pub exists: bool,
    pub eval: T,
    pub t_scaling: [T; 2],
}

impl<T: FloatLike> ESurfaceCacheTrait<T> for OneLoopESurfaceCache<T> {
    fn does_exist(&self) -> bool {
        utils::one_loop_e_surface_exists(&self.p1, &self.p2, self.m1_sq, self.m2_sq, self.e_shift)
    }

    fn eval(&self, k: &Vec<LorentzVector<T>>) -> T {
        let k_array = &[k[0].x, k[0].y, k[0].z];
        utils::one_loop_eval_e_surf(
            k_array,
            &self.p1,
            &self.p2,
            self.m1_sq,
            self.m2_sq,
            self.e_shift,
        )
    }

    #[inline]
    fn cached_eval(&self) -> T {
        self.eval
    }

    fn norm(&self, k: &Vec<LorentzVector<T>>) -> LorentzVector<T> {
        let k_array = &[k[0].x, k[0].y, k[0].z];
        let res = utils::one_loop_eval_e_surf_k_derivative(
            k_array, &self.p1, &self.p2, self.m1_sq, self.m2_sq,
        );
        LorentzVector {
            t: T::zero(),
            x: res[0],
            y: res[1],
            z: res[2],
        }
    }

    fn compute_t_scaling(&self, k: &Vec<LorentzVector<T>>) -> [T; 2] {
        let k_array = &[k[0].x, k[0].y, k[0].z];
        utils::one_loop_get_e_surf_t_scaling(
            k_array,
            &self.p1,
            &self.p2,
            self.m1_sq,
            self.m2_sq,
            self.e_shift,
        )
    }
}

#[allow(unused)]
impl<T: FloatLike> OneLoopESurfaceCache<T> {
    pub fn default() -> OneLoopESurfaceCache<T> {
        OneLoopESurfaceCache {
            p1: [T::zero(), T::zero(), T::zero()],
            p2: [T::zero(), T::zero(), T::zero()],
            m1_sq: T::zero(),
            m2_sq: T::zero(),
            e_shift: T::zero(),
            exists: true,
            eval: T::zero(),
            t_scaling: [T::zero(), T::zero()],
        }
    }

    pub fn new_from_inputs(
        p1: [T; 3],
        p2: [T; 3],
        m1_sq: T,
        m2_sq: T,
        e_shift: T,
    ) -> OneLoopESurfaceCache<T> {
        OneLoopESurfaceCache {
            p1: p1,
            p2: p2,
            m1_sq: m1_sq,
            m2_sq: m2_sq,
            e_shift: e_shift,
            exists: true,
            eval: T::zero(),
            t_scaling: [T::zero(), T::zero()],
        }
    }

    pub fn new(
        sigs: Vec<Vec<T>>,
        ps: Vec<[T; 3]>,
        ms: Vec<T>,
        e_shift: T,
        exists: bool,
        eval: T,
        t_scaling: [T; 2],
    ) -> GenericESurfaceCache<T> {
        // At one loop we require the momenta under both square roots to be normalised as k+p, so positive sig.
        assert!(sigs == vec![vec![T::one()], vec![T::one()]]);
        GenericESurfaceCache {
            sigs: sigs,
            ps: ps,
            ms: ms,
            e_shift: e_shift,
            exists: exists,
            eval: eval,
            t_scaling: t_scaling,
        }
    }

    pub fn bilinear_form(&self) -> ([[T; 3]; 3], [T; 3], T) {
        utils::one_loop_e_surface_bilinear_form(
            &self.p1,
            &self.p2,
            self.m1_sq,
            self.m2_sq,
            self.e_shift,
        )
    }
}

impl<T: FloatLike> ESurfaceCacheTrait<T> for GenericESurfaceCache<T> {
    fn does_exist(&self) -> bool {
        if self.ps.len() == 2 {
            utils::one_loop_e_surface_exists(
                &self.ps[0],
                &self.ps[1],
                self.ms[0],
                self.ms[1],
                self.e_shift,
            )
        } else if self.ps.len() == 3 {
            if self.e_shift > T::zero() {
                return false;
            }
            // We need to implement the basis change going from the following expression of the first two square roots:
            //  sqrt( (sig00 * k + sig01 * l + p0)^2 + m0^2) + sqrt( (sig10 * k + sig11 * l  + p1)^2 + m1^2 )
            // To:
            //  sqrt( k'^2 + m0^2 ) + sqrt( l'^2 + m1^2 )
            // Which we can achieve by the following basis change:
            //   k -> Inv[sig].k + (-Inv[sig].p)
            // With sig the matrix sig<ij> and k and p are two-vectors where each element are a 3-vector, i.e. k = {k,l}, p = {p0, p1}.
            //
            // We can then apply this basis transform to the third square root as well and read off the resulting
            // shift which we can use for the existence condition.
            //
            let s = self.sigs[0][0] * self.sigs[1][1] - self.sigs[0][1] * self.sigs[1][0];
            // Inverse signature matrix of the first two square roots
            let basis_change_matrix = [
                [s * self.sigs[1][1], s * self.sigs[0][1]],
                [s * self.sigs[1][0], s * self.sigs[0][0]],
            ];
            let basis_change_shifts = utils::two_loop_matrix_dot(
                basis_change_matrix,
                [
                    [-self.ps[0][0], -self.ps[0][1], -self.ps[0][2]],
                    [-self.ps[1][0], -self.ps[1][1], -self.ps[1][2]],
                ],
            );
            let defining_shift = [
                self.sigs[2][0] * basis_change_shifts[0][0]
                    + self.sigs[2][1] * basis_change_shifts[1][0]
                    + self.ps[2][0],
                self.sigs[2][0] * basis_change_shifts[0][1]
                    + self.sigs[2][1] * basis_change_shifts[1][1]
                    + self.ps[2][1],
                self.sigs[2][0] * basis_change_shifts[0][2]
                    + self.sigs[2][1] * basis_change_shifts[1][2]
                    + self.ps[2][2],
            ];
            let p_norm_sq = defining_shift[0] * defining_shift[0]
                + defining_shift[1] * defining_shift[1]
                + defining_shift[2] * defining_shift[2];
            let masses_sum = self.ms.iter().map(|m| m.sqrt()).sum::<T>();
            self.e_shift * self.e_shift - p_norm_sq > masses_sum * masses_sum
        } else {
            unimplemented!();
        }
    }

    #[inline]
    fn cached_eval(&self) -> T {
        self.eval
    }

    fn eval(&self, k: &Vec<LorentzVector<T>>) -> T {
        if self.ps.len() == 2 {
            let k_array = &[k[0].x, k[0].y, k[0].z];
            utils::one_loop_eval_e_surf(
                k_array,
                &self.ps[0],
                &self.ps[1],
                self.ms[0],
                self.ms[1],
                self.e_shift,
            )
        } else {
            let qs = self.compute_qs(k);
            self.ms
                .iter()
                .enumerate()
                .map(|(i, m_sq)| (qs[i].spatial_squared() + m_sq).sqrt())
                .sum::<T>()
                + self.e_shift
        }
    }

    fn norm(&self, k: &Vec<LorentzVector<T>>) -> LorentzVector<T> {
        if self.ps.len() == 2 {
            let k_array = &[k[0].x, k[0].y, k[0].z];
            let res = utils::one_loop_eval_e_surf_k_derivative(
                k_array,
                &self.ps[0],
                &self.ps[1],
                self.ms[0],
                self.ms[1],
            );
            LorentzVector {
                t: T::zero(),
                x: res[0],
                y: res[1],
                z: res[2],
            }
        } else {
            let mut norm = LorentzVector {
                t: T::zero(),
                x: T::zero(),
                y: T::zero(),
                z: T::zero(),
            };
            let qs = self.compute_qs(k);
            for (i_q, q) in qs.iter().enumerate() {
                norm += q / (q.spatial_squared() + self.ms[i_q]).sqrt();
            }
            norm
        }
    }

    fn compute_t_scaling(&self, k: &Vec<LorentzVector<T>>) -> [T; 2] {
        if self.ps.len() == 2 {
            let k_array = &[k[0].x, k[0].y, k[0].z];
            utils::one_loop_get_e_surf_t_scaling(
                k_array,
                &self.ps[0],
                &self.ps[1],
                self.ms[0],
                self.ms[1],
                self.e_shift,
            )
        } else {
            if self.ps.iter().all(|p| p.iter().all(|pi| *pi == T::zero())) {
                let qs = self.compute_qs(k);
                let t_sar = self.e_shift / qs.iter().map(|q| q.spatial_distance()).sum::<T>();
                [t_sar, -t_sar]
            } else {
                unimplemented!();
            }
        }
    }
}

#[allow(unused)]
impl<T: FloatLike> GenericESurfaceCache<T> {
    pub fn default() -> GenericESurfaceCache<T> {
        GenericESurfaceCache {
            sigs: vec![],
            ps: vec![],
            ms: vec![],
            e_shift: T::zero(),
            exists: true,
            eval: T::zero(),
            t_scaling: [T::zero(), T::zero()],
        }
    }

    pub fn new_from_inputs(
        sigs: Vec<Vec<T>>,
        ps: Vec<[T; 3]>,
        ms: Vec<T>,
        e_shift: T,
    ) -> GenericESurfaceCache<T> {
        // At one loop we require the momenta under both square roots to be normalised as k+p, so positive sig.
        if ps.len() == 2 {
            assert!(sigs == vec![vec![T::one()], vec![T::one()]]);
        }
        GenericESurfaceCache {
            sigs: sigs,
            ps: ps,
            ms: ms,
            e_shift: e_shift,
            exists: true,
            eval: T::zero(),
            t_scaling: [T::zero(), T::zero()],
        }
    }

    pub fn new(
        sigs: Vec<Vec<T>>,
        ps: Vec<[T; 3]>,
        ms: Vec<T>,
        e_shift: T,
        exists: bool,
        eval: T,
        t_scaling: [T; 2],
    ) -> GenericESurfaceCache<T> {
        if ps.len() == 2 {
            // At one loop we require the momenta under both square roots to be normalised as k+p, so positive sig.
            assert!(sigs == vec![vec![T::one()], vec![T::one()]]);
        }
        GenericESurfaceCache {
            sigs: sigs,
            ps: ps,
            ms: ms,
            e_shift: e_shift,
            exists: exists,
            eval: eval,
            t_scaling: t_scaling,
        }
    }

    pub fn adjust_loop_momenta_shifts(&mut self, loop_momenta_shift_adjustments: &Vec<[T; 3]>) {
        if loop_momenta_shift_adjustments.len() == 1 {
            self.ps[0][0] += loop_momenta_shift_adjustments[0][0];
            self.ps[0][1] += loop_momenta_shift_adjustments[0][1];
            self.ps[0][2] += loop_momenta_shift_adjustments[0][2];

            self.ps[1][0] += loop_momenta_shift_adjustments[0][0];
            self.ps[1][1] += loop_momenta_shift_adjustments[0][1];
            self.ps[1][2] += loop_momenta_shift_adjustments[0][2];
        } else {
            for (i, p) in self.ps.iter_mut().enumerate() {
                for (j, shift) in loop_momenta_shift_adjustments.iter().enumerate() {
                    p[0] += self.sigs[i][j] * shift[0];
                    p[1] += self.sigs[i][j] * shift[1];
                    p[2] += self.sigs[i][j] * shift[2];
                }
            }
        }
    }

    pub fn compute_qs(&self, k: &Vec<LorentzVector<T>>) -> Vec<LorentzVector<T>> {
        let mut qs = vec![];
        for ss in self.sigs.iter() {
            let mut new_vec = LorentzVector {
                t: T::zero(),
                x: T::zero(),
                y: T::zero(),
                z: T::zero(),
            };
            for (i_s, s) in ss.iter().enumerate() {
                new_vec += k[i_s] * (*s);
            }
            qs.push(new_vec);
        }
        qs
    }

    pub fn bilinear_form(&self) -> ([[T; 3]; 3], [T; 3], T) {
        if self.ps.len() == 2 {
            utils::one_loop_e_surface_bilinear_form(
                &self.ps[0],
                &self.ps[1],
                self.ms[0],
                self.ms[1],
                self.e_shift,
            )
        } else {
            unimplemented!();
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct CFFFactor {
    // Denominator consists of a list of E-surface IDs appearing in the denominator
    pub denominator: Vec<usize>,
    pub factors: Vec<CFFFactor>,
}
#[allow(unused)]
impl CFFFactor {
    pub fn new(denominator: Vec<usize>, factors: Vec<CFFFactor>) -> CFFFactor {
        CFFFactor {
            denominator,
            factors,
        }
    }

    pub fn evaluate<T: FloatLike, ESC: ESurfaceCacheTrait<T>>(
        &self,
        e_surface_caches: &Vec<ESC>,
        expand_e_surf: Option<(usize, T)>,
        has_found_expanded_e_surf: bool,
    ) -> T {
        let mut coef = T::one();
        let mut next_has_found_expanded_e_surf = has_found_expanded_e_surf;
        for e_surf_id in self.denominator.iter() {
            if let Some((expanded_e_surf_id, expanded_e_surf)) = expand_e_surf {
                if expanded_e_surf_id != *e_surf_id {
                    coef *= e_surface_caches[*e_surf_id].cached_eval();
                } else {
                    if has_found_expanded_e_surf {
                        panic!("Current implementation only supports E-surfaces that appear as single-pole only.");
                    }
                    next_has_found_expanded_e_surf = true;
                    coef *= expanded_e_surf;
                }
            } else {
                coef *= e_surface_caches[*e_surf_id].cached_eval();
            }
        }
        if self.factors.len() > 0 {
            let mut factors_sum = T::zero();
            for factor in self.factors.iter() {
                factors_sum += factor.evaluate(
                    e_surface_caches,
                    expand_e_surf,
                    next_has_found_expanded_e_surf,
                );
            }
            coef.inv() * factors_sum
        } else {
            if expand_e_surf.is_some() && !next_has_found_expanded_e_surf {
                T::zero()
            } else {
                coef.inv()
            }
        }
    }

    pub fn contains_e_surf_id(&self, e_surf_id: usize) -> bool {
        if self.denominator.contains(&e_surf_id) {
            true
        } else {
            for factor in self.factors.iter() {
                if factor.contains_e_surf_id(e_surf_id) {
                    return true;
                }
            }
            false
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct CFFTerm {
    pub orientation: Vec<(usize, isize)>,
    pub factors: Vec<CFFFactor>,
}
#[allow(unused)]
impl CFFTerm {
    pub fn new(orientation: Vec<(usize, isize)>, factors: Vec<CFFFactor>) -> CFFTerm {
        CFFTerm {
            orientation,
            factors,
        }
    }

    pub fn evaluate<T: FloatLike, ESC: ESurfaceCacheTrait<T>>(
        &self,
        e_surface_caches: &Vec<ESC>,
        expand_e_surf: Option<(usize, T)>,
    ) -> T {
        let mut result = T::zero();
        for factor in self.factors.iter() {
            result += factor.evaluate(e_surface_caches, expand_e_surf, false);
        }
        result
    }

    pub fn contains_e_surf_id(&self, e_surf_id: usize) -> bool {
        for factor in self.factors.iter() {
            if factor.contains_e_surf_id(e_surf_id) {
                return true;
            }
        }
        return false;
    }
}
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CFFExpression {
    pub terms: Vec<CFFTerm>,
    pub e_surfaces: Vec<Esurface>,
}
#[allow(unused)]
impl CFFExpression {
    pub fn new(terms: Vec<CFFTerm>, e_surfaces: Vec<Esurface>) -> CFFExpression {
        CFFExpression { terms, e_surfaces }
    }
}
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Amplitude {
    pub edges: Vec<Edge>,
    pub lmb_edges: Vec<Edge>,
    pub external_edge_id_and_flip: Vec<(isize, isize)>,
    pub cff_expression: CFFExpression,
    pub n_loop: usize,
}
#[allow(unused)]
impl Amplitude {
    pub fn new(
        edges: Vec<Edge>,
        lmb_edges: Vec<Edge>,
        external_edge_id_and_flip: Vec<(isize, isize)>,
        cff_expression: CFFExpression,
        thresholds: Vec<usize>,
        n_loop: usize,
    ) -> Amplitude {
        Amplitude {
            edges,
            lmb_edges,
            external_edge_id_and_flip,
            cff_expression,
            n_loop,
        }
    }
}
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Cut {
    pub cut_edge_ids_and_flip: Vec<(usize, isize)>,
    pub left_amplitude: Amplitude,
    pub right_amplitude: Amplitude,
}
#[allow(unused)]
impl Cut {
    pub fn new(
        cut_edge_ids_and_flip: Vec<(usize, isize)>,
        left_amplitude: Amplitude,
        right_amplitude: Amplitude,
    ) -> Cut {
        Cut {
            cut_edge_ids_and_flip,
            left_amplitude,
            right_amplitude,
        }
    }
}
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Edge {
    pub mass: f64,
    pub signature: (Vec<isize>, Vec<isize>),
    pub id: usize,
    pub power: usize,
}
#[allow(unused)]
impl Edge {
    pub fn new(mass: f64, signature: (Vec<isize>, Vec<isize>), power: usize, id: usize) -> Edge {
        Edge {
            mass,
            signature,
            id,
            power,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SuperGraph {
    pub edges: Vec<Edge>,
    pub cuts: Vec<Cut>,
    pub n_loop: usize,
}
#[allow(unused)]
impl SuperGraph {
    pub fn new(edges: Vec<Edge>, cuts: Vec<Cut>, n_loop: usize) -> SuperGraph {
        SuperGraph {
            edges,
            cuts,
            n_loop,
        }
    }

    pub fn default() -> SuperGraph {
        SuperGraph {
            edges: Vec::new(),
            cuts: Vec::new(),
            n_loop: 0,
        }
    }

    pub fn from_file(filename: &str) -> Result<SuperGraph, Report> {
        let f = File::open(filename)
            .wrap_err_with(|| format!("Could not open supergraph file {}", filename))
            .suggestion("Does the path exist?")?;
        serde_yaml::from_reader(f)
            .wrap_err("Could not parse supergraph file")
            .suggestion("Is it a correct YAML file")
    }
}

#[enum_dispatch]
pub trait HasIntegrand {
    fn create_grid(&self) -> Grid;

    fn evaluate_sample(
        &self,
        sample: &Sample,
        wgt: f64,
        iter: usize,
        use_f128: bool,
    ) -> Complex<f64>;

    fn get_n_dim(&self) -> usize;
}

#[enum_dispatch(HasIntegrand)]
pub enum Integrand {
    UnitSurface(UnitSurfaceIntegrand),
    UnitVolume(UnitVolumeIntegrand),
    HFunctionTest(HFunctionTestIntegrand),
    LoopInducedTriBoxTri(LoopInducedTriBoxTriIntegrand),
    TriBoxTri(TriBoxTriIntegrand),
    TriangleSubtraction(TriangleSubtractionIntegrand),
    BoxSubtraction(BoxSubtractionIntegrand),
}

pub fn integrand_factory(settings: &Settings) -> Integrand {
    match settings.hard_coded_integrand.clone() {
        HardCodedIntegrandSettings::UnitSurface(integrand_settings) => Integrand::UnitSurface(
            UnitSurfaceIntegrand::new(settings.clone(), integrand_settings),
        ),
        HardCodedIntegrandSettings::UnitVolume(integrand_settings) => Integrand::UnitVolume(
            UnitVolumeIntegrand::new(settings.clone(), integrand_settings),
        ),
        HardCodedIntegrandSettings::HFunctionTest(integrand_settings) => Integrand::HFunctionTest(
            HFunctionTestIntegrand::new(settings.clone(), integrand_settings),
        ),
        HardCodedIntegrandSettings::LoopInducedTriBoxTri(integrand_settings) => {
            Integrand::LoopInducedTriBoxTri(LoopInducedTriBoxTriIntegrand::new(
                settings.clone(),
                integrand_settings,
            ))
        }
        HardCodedIntegrandSettings::TriBoxTri(integrand_settings) => Integrand::TriBoxTri(
            TriBoxTriIntegrand::new(settings.clone(), integrand_settings),
        ),
        HardCodedIntegrandSettings::TriangleSubtraction(integrand_settings) => {
            Integrand::TriangleSubtraction(TriangleSubtractionIntegrand::new(integrand_settings))
        }
        HardCodedIntegrandSettings::BoxSubtraction(integrand_settings) => {
            Integrand::BoxSubtraction(BoxSubtractionIntegrand::new(integrand_settings))
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct UnitSurfaceSettings {
    pub n_3d_momenta: usize,
}

pub struct UnitSurfaceIntegrand {
    pub settings: Settings,
    pub n_dim: usize,
    pub n_3d_momenta: usize,
    pub supergraph: SuperGraph,
    pub surface: f64,
}

#[allow(unused)]
impl UnitSurfaceIntegrand {
    pub fn new(
        settings: Settings,
        integrand_settings: UnitSurfaceSettings,
    ) -> UnitSurfaceIntegrand {
        let n_dim =
            utils::get_n_dim_for_n_loop_momenta(&settings, integrand_settings.n_3d_momenta, true);
        let surface = utils::compute_surface_and_volume(
            integrand_settings.n_3d_momenta * 3 - 1,
            settings.kinematics.e_cm,
        )
        .0;
        UnitSurfaceIntegrand {
            settings,
            n_3d_momenta: integrand_settings.n_3d_momenta,
            n_dim: n_dim,
            supergraph: SuperGraph::default(),
            surface: surface,
        }
    }

    fn evaluate_numerator<T: FloatLike>(&self, loop_momenta: &[LorentzVector<T>]) -> T {
        return T::from_f64(1.0).unwrap();
    }

    fn parameterize<T: FloatLike>(&self, xs: &[T]) -> (Vec<[T; 3]>, T) {
        utils::global_parameterize(
            xs,
            Into::<T>::into(self.settings.kinematics.e_cm * self.settings.kinematics.e_cm),
            &self.settings,
            true,
        )
    }
}

#[allow(unused)]
impl HasIntegrand for UnitSurfaceIntegrand {
    fn create_grid(&self) -> Grid {
        Grid::ContinuousGrid(ContinuousGrid::new(
            self.n_dim,
            self.settings.integrator.n_bins,
            self.settings.integrator.min_samples_for_update,
        ))
    }

    fn get_n_dim(&self) -> usize {
        return self.n_dim;
    }

    fn evaluate_sample(
        &self,
        sample: &Sample,
        wgt: f64,
        iter: usize,
        use_f128: bool,
    ) -> Complex<f64> {
        let xs = match sample {
            Sample::ContinuousGrid(_w, v) => v,
            _ => panic!("Wrong sample type"),
        };
        let mut sample_xs = vec![self.settings.kinematics.e_cm];
        sample_xs.extend(xs);
        let (moms, jac) = self.parameterize(sample_xs.as_slice());
        let mut loop_momenta = vec![];
        for m in &moms {
            loop_momenta.push(LorentzVector {
                t: ((m[0] + m[1] + m[2]) * (m[0] + m[1] + m[2])).sqrt(),
                x: m[0],
                y: m[1],
                z: m[2],
            });
        }
        let mut itg_wgt = self.evaluate_numerator(loop_momenta.as_slice());
        // Normalize the integral
        itg_wgt /= self.surface;
        if self.settings.general.debug > 1 {
            println!("Sampled loop momenta:");
            for (i, l) in loop_momenta.iter().enumerate() {
                println!(
                    "k{} = ( {:-23}, {:-23}, {:-23}, {:-23} )",
                    i,
                    format!("{:+.16e}", l.t),
                    format!("{:+.16e}", l.x),
                    format!("{:+.16e}", l.y),
                    format!("{:+.16e}", l.z)
                );
            }
            println!("Integrator weight : {:+.16e}", wgt);
            println!("Integrand weight  : {:+.16e}", itg_wgt);
            println!("Sampling jacobian : {:+.16e}", jac);
            println!("Final contribution: {:+.16e}", itg_wgt * jac);
        }
        return Complex::new(itg_wgt, 0.) * jac;
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct UnitVolumeSettings {
    pub n_3d_momenta: usize,
}

pub struct UnitVolumeIntegrand {
    pub settings: Settings,
    pub n_dim: usize,
    pub n_3d_momenta: usize,
    pub supergraph: SuperGraph,
    pub volume: f64,
}

#[allow(unused)]
impl UnitVolumeIntegrand {
    pub fn new(settings: Settings, integrand_settings: UnitVolumeSettings) -> UnitVolumeIntegrand {
        let n_dim =
            utils::get_n_dim_for_n_loop_momenta(&settings, integrand_settings.n_3d_momenta, false);
        let volume = utils::compute_surface_and_volume(
            integrand_settings.n_3d_momenta * 3,
            settings.kinematics.e_cm,
        )
        .1;
        UnitVolumeIntegrand {
            settings,
            n_3d_momenta: integrand_settings.n_3d_momenta,
            n_dim: n_dim,
            supergraph: SuperGraph::default(),
            volume: volume,
        }
    }

    fn evaluate_numerator<T: FloatLike>(&self, loop_momenta: &[LorentzVector<T>]) -> T {
        if loop_momenta
            .iter()
            .map(|l| l.spatial_squared())
            .sum::<T>()
            .sqrt()
            > Into::<T>::into(self.settings.kinematics.e_cm)
        {
            return T::from_f64(0.0).unwrap();
        } else {
            return T::from_f64(1.0).unwrap();
        }
    }

    fn parameterize<T: FloatLike>(&self, xs: &[T]) -> (Vec<[T; 3]>, T) {
        utils::global_parameterize(
            xs,
            Into::<T>::into(self.settings.kinematics.e_cm * self.settings.kinematics.e_cm),
            &self.settings,
            false,
        )
    }
}

#[allow(unused)]
impl HasIntegrand for UnitVolumeIntegrand {
    fn create_grid(&self) -> Grid {
        Grid::ContinuousGrid(ContinuousGrid::new(
            self.n_dim,
            self.settings.integrator.n_bins,
            self.settings.integrator.min_samples_for_update,
        ))
    }

    fn get_n_dim(&self) -> usize {
        return self.n_dim;
    }

    fn evaluate_sample(
        &self,
        sample: &Sample,
        wgt: f64,
        iter: usize,
        use_f128: bool,
    ) -> Complex<f64> {
        let xs = match sample {
            Sample::ContinuousGrid(_w, v) => v,
            _ => panic!("Wrong sample type"),
        };
        let (moms, jac) = self.parameterize(xs);
        let mut loop_momenta = vec![];
        for m in &moms {
            loop_momenta.push(LorentzVector {
                t: 0.,
                x: m[0],
                y: m[1],
                z: m[2],
            });
        }
        let mut itg_wgt = self.evaluate_numerator(loop_momenta.as_slice());
        // Normalize the integral
        itg_wgt /= self.volume;
        if self.settings.general.debug > 1 {
            println!("Sampled loop momenta:");
            for (i, l) in loop_momenta.iter().enumerate() {
                println!(
                    "k{} = ( {:-23}, {:-23}, {:-23}, {:-23} )",
                    i,
                    format!("{:+.16e}", l.t),
                    format!("{:+.16e}", l.x),
                    format!("{:+.16e}", l.y),
                    format!("{:+.16e}", l.z)
                );
            }
            println!("Integrator weight : {:+.16e}", wgt);
            println!("Integrand weight  : {:+.16e}", itg_wgt);
            println!("Sampling jacobian : {:+.16e}", jac);
            println!("Final contribution: {:+.16e}", itg_wgt * jac);
        }
        return Complex::new(itg_wgt, 0.) * jac;
    }
}
