use crate::{ParameterizationMapping, ParameterizationMode, Settings, MAX_LOOP};
use hyperdual::Hyperdual;
use itertools::{izip, Itertools};
use lorentz_vector::{Field, LorentzVector, RealNumberLike};
use num::Complex;
use num::ToPrimitive;
use num_traits::{Float, FloatConst, FromPrimitive, Num, NumAssign, NumCast, Signed};
use num_traits::{Inv, One, Zero};
use std::cmp::{Ord, Ordering};
use std::ops::Neg;

#[allow(unused)]
const MAX_DIMENSION: usize = MAX_LOOP * 3;

pub trait FloatConvertFrom<U> {
    fn convert_from(x: &U) -> Self;
}

impl FloatConvertFrom<f64> for f64 {
    fn convert_from(x: &f64) -> f64 {
        *x
    }
}

impl FloatConvertFrom<f128::f128> for f64 {
    fn convert_from(x: &f128::f128) -> f64 {
        (*x).to_f64().unwrap()
    }
}

impl FloatConvertFrom<f128::f128> for f128::f128 {
    fn convert_from(x: &f128::f128) -> f128::f128 {
        *x
    }
}

impl FloatConvertFrom<f64> for f128::f128 {
    fn convert_from(x: &f64) -> f128::f128 {
        f128::f128::from_f64(*x).unwrap()
    }
}

pub trait FloatLike:
    From<f64>
    + FloatConvertFrom<f64>
    + FloatConvertFrom<f128::f128>
    + Num
    + FromPrimitive
    + Float
    + Field
    + RealNumberLike
    + Signed
    + FloatConst
    + std::fmt::LowerExp
    + 'static
    + Signum
{
}

impl FloatLike for f64 {}
impl FloatLike for f128::f128 {}

/// An iterator which iterates two other iterators simultaneously
#[derive(Clone, Debug)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct ZipEq<I, J> {
    a: I,
    b: J,
}

/// An iterator which iterates two other iterators simultaneously and checks
/// if the sizes are equal in debug mode.
#[allow(unused)]
pub fn zip_eq<I, J>(i: I, j: J) -> ZipEq<I::IntoIter, J::IntoIter>
where
    I: IntoIterator,
    J: IntoIterator,
{
    ZipEq {
        a: i.into_iter(),
        b: j.into_iter(),
    }
}

impl<I, J> Iterator for ZipEq<I, J>
where
    I: Iterator,
    J: Iterator,
{
    type Item = (I::Item, J::Item);

    fn next(&mut self) -> Option<Self::Item> {
        match (self.a.next(), self.b.next()) {
            (None, None) => None,
            (Some(a), Some(b)) => Some((a, b)),
            (None, Some(_)) => {
                #[cfg(debug_assertions)]
                panic!("Unequal length of iterators; first iterator finished first");
                #[cfg(not(debug_assertions))]
                None
            }
            (Some(_), None) => {
                #[cfg(debug_assertions)]
                panic!("Unequal length of iterators; second iterator finished first");
                #[cfg(not(debug_assertions))]
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let sa = self.a.size_hint();
        let sb = self.b.size_hint();
        (sa.0.min(sb.0), sa.1.zip(sb.1).map(|(ua, ub)| ua.min(ub)))
    }
}

impl<I, J> ExactSizeIterator for ZipEq<I, J>
where
    I: ExactSizeIterator,
    J: ExactSizeIterator,
{
}

/// Format a mean ± sdev as mean(sdev) with the correct number of digits.
/// Based on the Python package gvar.
pub fn format_uncertainty(mean: f64, sdev: f64) -> String {
    fn ndec(x: f64, offset: usize) -> i32 {
        let mut ans = (offset as f64 - x.log10()) as i32;
        if ans > 0 && x * 10.0.powi(ans) >= [0.5, 9.5, 99.5][offset] {
            ans -= 1;
        }
        if ans < 0 {
            0
        } else {
            ans
        }
    }
    let v = mean;
    let dv = sdev.abs();

    // special cases
    if v.is_nan() || dv.is_nan() {
        format!("{:e} ± {:e}", v, dv)
    } else if dv.is_infinite() {
        format!("{:e} ± inf", v)
    } else if v == 0. && (dv >= 1e5 || dv < 1e-4) {
        if dv == 0. {
            "0(0)".to_owned()
        } else {
            let e = format!("{:.1e}", dv);
            let mut ans = e.split('e');
            let e1 = ans.next().unwrap();
            let e2 = ans.next().unwrap();
            "0.0(".to_owned() + e1 + ")e" + e2
        }
    } else if v == 0. {
        if dv >= 9.95 {
            format!("0({:.0})", dv)
        } else if dv >= 0.995 {
            format!("0.0({:.1})", dv)
        } else {
            let ndecimal = ndec(dv, 2);
            format!(
                "{:.*}({:.0})",
                ndecimal as usize,
                v,
                dv * 10.0.powi(ndecimal)
            )
        }
    } else if dv == 0. {
        let e = format!("{:e}", v);
        let mut ans = e.split('e');
        let e1 = ans.next().unwrap();
        let e2 = ans.next().unwrap();
        if e2 != "0" {
            e1.to_owned() + "(0)e" + e2
        } else {
            e1.to_owned() + "(0)"
        }
    } else if dv > 1e4 * v.abs() {
        format!("{:.1e} ± {:.2e}", v, dv)
    } else if v.abs() >= 1e6 || v.abs() < 1e-5 {
        // exponential notation for large |self.mean|
        let exponent = v.abs().log10().floor();
        let fac = 10.0.powf(exponent);
        let mantissa = format_uncertainty(v / fac, dv / fac);
        let e = format!("{:.0e}", fac);
        let mut ee = e.split('e');
        mantissa + "e" + ee.nth(1).unwrap()
    }
    // normal cases
    else if dv >= 9.95 {
        if v.abs() >= 9.5 {
            format!("{:.0}({:.0})", v, dv)
        } else {
            let ndecimal = ndec(v.abs(), 1);
            format!("{:.*}({:.*})", ndecimal as usize, v, ndecimal as usize, dv)
        }
    } else if dv >= 0.995 {
        if v.abs() >= 0.95 {
            format!("{:.1}({:.1})", v, dv)
        } else {
            let ndecimal = ndec(v.abs(), 1);
            format!("{:.*}({:.*})", ndecimal as usize, v, ndecimal as usize, dv)
        }
    } else {
        let ndecimal = ndec(v.abs(), 1).max(ndec(dv, 2));
        format!(
            "{:.*}({:.0})",
            ndecimal as usize,
            v,
            dv * 10.0.powi(ndecimal)
        )
    }
}

/// Compare two slices, selecting on length first
#[allow(unused)]
pub fn compare_slice<T: Ord>(slice1: &[T], slice2: &[T]) -> Ordering {
    match slice1.len().cmp(&slice2.len()) {
        Ordering::Equal => (),
        non_eq => return non_eq,
    }

    let l = slice1.len();
    // Slice to the loop iteration range to enable bound check
    // elimination in the compiler
    let lhs = &slice1[..l];
    let rhs = &slice2[..l];

    for i in 0..l {
        match lhs[i].cmp(&rhs[i]) {
            Ordering::Equal => (),
            non_eq => return non_eq,
        }
    }

    Ordering::Equal
}

pub trait Signum {
    fn multiply_sign(&self, sign: i8) -> Self;
}

impl Signum for f128::f128 {
    #[inline]
    fn multiply_sign(&self, sign: i8) -> f128::f128 {
        match sign {
            1 => self.clone(),
            0 => f128::f128::zero(),
            -1 => self.neg(),
            _ => unreachable!("Sign should be -1,0,1"),
        }
    }
}

impl Signum for f64 {
    #[inline]
    fn multiply_sign(&self, sign: i8) -> f64 {
        match sign {
            1 => self.clone(),
            0 => f64::zero(),
            -1 => self.neg(),
            _ => unreachable!("Sign should be -1,0,1"),
        }
    }
}

impl<T: Num + Neg<Output = T> + Copy> Signum for Complex<T> {
    #[inline]
    fn multiply_sign(&self, sign: i8) -> Complex<T> {
        match sign {
            1 => *self,
            0 => Complex::zero(),
            -1 => Complex::new(-self.re, -self.im),
            _ => unreachable!("Sign should be -1,0,1"),
        }
    }
}

impl<T: FloatLike, const U: usize> Signum for Hyperdual<T, U> {
    #[inline]
    fn multiply_sign(&self, sign: i8) -> Hyperdual<T, U> {
        match sign {
            1 => *self,
            0 => Hyperdual::zero(),
            -1 => -*self,
            _ => unreachable!("Sign should be -1,0,1"),
        }
    }
}

impl<T: Field> Signum for LorentzVector<T> {
    #[inline]
    fn multiply_sign(&self, sign: i8) -> LorentzVector<T> {
        match sign {
            1 => *self,
            0 => LorentzVector::default(),
            -1 => -self,
            _ => unreachable!("Sign should be -1,0,1"),
        }
    }
}

#[allow(unused)]
#[inline]
/// Invert with better precision
pub fn finv<T: Float>(c: Complex<T>) -> Complex<T> {
    let norm = c.norm();
    c.conj() / norm / norm
}

#[allow(unused)]
#[inline]
pub fn powi<T: Float + NumAssign>(c: Complex<T>, n: usize) -> Complex<T> {
    let mut c1 = Complex::<T>::one();
    for _ in 0..n {
        c1 *= c;
    }
    c1
}

#[allow(unused)]
#[inline]
pub fn powf<T: 'static + Float + NumAssign + std::fmt::Debug, const N: usize>(
    h: Hyperdual<T, N>,
    n: T,
) -> Hyperdual<T, N> {
    let r = Float::powf(h.real(), n - T::one());
    let rr = n * r;
    h.map_dual(r * h.real(), |x| *x * rr)
}

#[allow(unused)]
pub fn evaluate_signature<T: Field>(
    signature: &[i8],
    momenta: &[LorentzVector<T>],
) -> LorentzVector<T> {
    let mut momentum = LorentzVector::default();
    for (&sign, mom) in zip_eq(signature, momenta) {
        match sign {
            0 => {}
            1 => momentum += mom,
            -1 => momentum -= mom,
            _ => {
                #[cfg(debug_assertions)]
                panic!("Sign should be -1,0,1")
            }
        }
    }

    momentum
}

/// Calculate the determinant of any complex-valued input matrix using LU-decomposition.
/// Original C-code by W. Gong and D.E. Soper.
#[allow(unused)]
pub fn determinant<T: Float + RealNumberLike>(
    bb: &Vec<Complex<T>>,
    dimension: usize,
) -> Complex<T> {
    // Define matrix related variables.
    let mut determinant = Complex::new(T::one(), T::zero());
    let mut indx = [0; MAX_DIMENSION];
    let mut d = 1; // initialize parity parameter

    // Inintialize the matrix to be decomposed with the transferred matrix b.
    let mut aa = bb.clone();

    // Define parameters used in decomposition.
    let mut imax = 0;
    let mut flag = 1;
    let mut dumc;
    let mut sum;

    let mut aamax;
    let mut dumr;
    let mut vv = [T::zero(); MAX_DIMENSION];

    // Get the implicit scaling information.
    for i in 0..dimension {
        aamax = T::zero();
        for j in 0..dimension {
            let r = aa[i * dimension + j].norm_sqr();
            if r > aamax {
                aamax = r;
            }
        }
        // Set a flag to check if the determinant is zero.
        if aamax.is_zero() {
            flag = 0;
        }
        // Save the scaling.
        vv[i] = aamax.inv();
    }
    if flag == 1 {
        for j in 0..dimension {
            for i in 0..j {
                sum = aa[i * dimension + j];
                for k in 0..i {
                    sum = sum - aa[i * dimension + k] * aa[k * dimension + j];
                }
                aa[i * dimension + j] = sum;
            }
            //Initialize for the search for largest pivot element.
            aamax = T::zero();
            for i in j..dimension {
                sum = aa[i * dimension + j];
                for k in 0..j {
                    sum = sum - aa[i * dimension + k] * aa[k * dimension + j];
                }
                aa[i * dimension + j] = sum;
                // Figure of merit for the pivot.
                dumr = vv[i] * sum.norm_sqr();
                // Is it better than the best so far?
                if dumr >= aamax {
                    imax = i;
                    aamax = dumr;
                }
            }
            // See if we need to interchange rows.
            if j != imax {
                for k in 0..dimension {
                    dumc = aa[imax * dimension + k];
                    aa[imax * dimension + k] = aa[j * dimension + k];
                    aa[j * dimension + k] = dumc;
                }
                // Change the parity of d.
                d = -d;
                // Interchange the scale factor.
                vv[imax] = vv[j];
            }
            indx[j] = imax;
            if j + 1 != dimension {
                dumc = aa[j * dimension + j].inv();
                for i in j + 1..dimension {
                    aa[i * dimension + j] = aa[i * dimension + j] * dumc;
                }
            }
        }
    }
    // Calculate the determinant using the decomposed matrix.
    if flag == 0 {
        determinant = Complex::default();
    } else {
        // Multiply the diagonal elements.
        for diagonal in 0..dimension {
            determinant = determinant * aa[diagonal * dimension + diagonal];
        }
        determinant = determinant * <T as NumCast>::from(d).unwrap();
    }
    determinant
}

#[allow(unused)]
pub fn next_combination_with_replacement(state: &mut [usize], max_entry: usize) -> bool {
    for i in (0..state.len()).rev() {
        if state[i] < max_entry {
            state[i] += 1;
            for j in i + 1..state.len() {
                state[j] = state[i]
            }
            return true;
        }
    }
    false
}

pub fn global_parameterize<T: FloatLike>(
    x: &[T],
    e_cm_squared: T,
    settings: &Settings,
    force_radius: bool,
) -> (Vec<[T; 3]>, T) {
    match settings.parameterization.mode {
        ParameterizationMode::HyperSpherical => {
            let e_cm = e_cm_squared.sqrt() * Into::<T>::into(settings.parameterization.shifts[0].0);
            let mut jac = T::one();
            // rescale the input to the desired range
            let mut x_r = Vec::with_capacity(x.len());
            if !force_radius {
                x_r.push(x[0]);
            } else {
                let lo = Into::<T>::into(settings.parameterization.input_rescaling[0][0].0);
                let hi = Into::<T>::into(settings.parameterization.input_rescaling[0][0].1);
                x_r.push(lo + x[0] * (hi - lo));
                jac *= Into::<T>::into(hi - lo);
            }
            let lo = Into::<T>::into(settings.parameterization.input_rescaling[0][1].0);
            let hi = Into::<T>::into(settings.parameterization.input_rescaling[0][1].1);
            x_r.push(lo + x[1] * (hi - lo));
            jac *= Into::<T>::into(hi - lo);
            for xi in &x[2..] {
                let lo = Into::<T>::into(settings.parameterization.input_rescaling[0][2].0);
                let hi = Into::<T>::into(settings.parameterization.input_rescaling[0][2].1);
                x_r.push(lo + *xi * (hi - lo));
                jac *= Into::<T>::into(hi - lo);
            }

            let radius: T = if force_radius {
                x[0]
            } else {
                match settings.parameterization.mapping {
                    ParameterizationMapping::Log => {
                        // r = e_cm * ln(1 + b*x/(1-x))
                        let x = x_r[0];
                        let b = Into::<T>::into(settings.parameterization.b);
                        let radius = e_cm * (T::one() + b * x / (T::one() - x)).ln();
                        jac *= e_cm * b / (T::one() - x) / (T::one() + x * (b - T::one()));
                        radius
                    }
                    ParameterizationMapping::Linear => {
                        // r = e_cm * b * x/(1-x)
                        let b = Into::<T>::into(settings.parameterization.b);
                        let radius = e_cm * b * x_r[0] / (T::one() - x_r[0]);
                        jac *= <T as num_traits::Float>::powi(e_cm * b + radius, 2) / e_cm / b;
                        radius
                    }
                }
            };

            let phi = Into::<T>::into(2.) * <T as FloatConst>::PI() * x_r[1];
            jac *= Into::<T>::into(2.) * <T as FloatConst>::PI();

            let mut cos_thetas = Vec::with_capacity(x.len() - 2);
            let mut sin_thetas = Vec::with_capacity(x.len() - 2);

            for (i, xi) in x_r[2..x_r.len()].iter().enumerate() {
                let cos_theta = -T::one() + Into::<T>::into(2.) * *xi;
                jac *= Into::<T>::into(2.);
                let sin_theta = (T::one() - cos_theta * cos_theta).sqrt();
                if i > 0 {
                    jac *= sin_theta.powi(i as i32);
                }
                cos_thetas.push(cos_theta);
                sin_thetas.push(sin_theta);
            }

            let mut concatenated_vecs = Vec::with_capacity(x.len() / 3);
            let mut base = radius;
            for (_i, (cos_theta, sin_theta)) in cos_thetas.iter().zip(sin_thetas.iter()).enumerate()
            {
                concatenated_vecs.push(base * cos_theta);
                base *= *sin_theta;
            }
            concatenated_vecs.push(base * phi.cos());
            concatenated_vecs.push(base * phi.sin());

            jac *= radius.powi((x.len() - 1) as i32); // hyperspherical coords

            (
                concatenated_vecs
                    .chunks(3)
                    .map(|v| [v[0], v[1], v[2]])
                    .collect(),
                jac,
            )
        }
        _ => {
            if force_radius {
                panic!("Cannot force radius for non-hyper-spherical parameterization.");
            }
            let mut jac = T::one();
            let mut vecs = Vec::with_capacity(x.len() / 3);
            for (i, xi) in x.chunks(3).enumerate() {
                let (vec_i, jac_i) = parameterize3d(xi, e_cm_squared, i, settings);
                vecs.push(vec_i);
                jac *= jac_i;
            }
            (vecs, jac)
        }
    }
}

#[allow(unused)]
pub fn global_inv_parameterize<T: FloatLike>(
    moms: &Vec<LorentzVector<T>>,
    e_cm_squared: T,
    settings: &Settings,
    force_radius: bool,
) -> (Vec<T>, T) {
    match settings.parameterization.mode {
        ParameterizationMode::HyperSpherical => {
            let e_cm = e_cm_squared.sqrt() * Into::<T>::into(settings.parameterization.shifts[0].0);
            let mut inv_jac = T::one();
            let mut xs = Vec::with_capacity(moms.len() * 3);

            let cartesian_xs = moms
                .iter()
                .map(|lv| [lv.x, lv.y, lv.z])
                .flatten()
                .collect::<Vec<_>>();

            let mut k_r_sq = cartesian_xs.iter().map(|xi| *xi * xi).sum::<T>();
            // cover the degenerate case
            if k_r_sq.is_zero() {
                return (vec![T::zero(); cartesian_xs.len()], T::zero());
            }
            let k_r = k_r_sq.sqrt();
            if force_radius {
                xs.push(k_r);
            } else {
                match settings.parameterization.mapping {
                    ParameterizationMapping::Log => {
                        let b = Into::<T>::into(settings.parameterization.b);
                        let x1 = T::one() - b / (-T::one() + b + (k_r / e_cm).exp());
                        inv_jac /= e_cm * b / (T::one() - x1) / (T::one() + x1 * (b - T::one()));
                        xs.push(x1);
                    }
                    ParameterizationMapping::Linear => {
                        let b = Into::<T>::into(settings.parameterization.b);
                        inv_jac /= <T as num_traits::Float>::powi(e_cm * b + k_r, 2) / e_cm / b;
                        xs.push(k_r / (e_cm * b + k_r));
                    }
                }
            };

            let y = cartesian_xs[cartesian_xs.len() - 2];
            let x = cartesian_xs[cartesian_xs.len() - 1];
            let xphi = if x < T::zero() {
                T::one() + Into::<T>::into(0.5) * T::FRAC_1_PI() * T::atan2(x, y)
            } else {
                Into::<T>::into(0.5) * T::FRAC_1_PI() * T::atan2(x, y)
            };
            xs.push(xphi);
            inv_jac /= Into::<T>::into(2.) * <T as FloatConst>::PI();

            for x in &cartesian_xs[..cartesian_xs.len() - 2] {
                xs.push(Into::<T>::into(0.5) * (T::one() + *x / k_r_sq.sqrt()));
                k_r_sq -= *x * x;
                inv_jac /= Into::<T>::into(2.);
                //TODO implement the 1/sin^i(theta) term
            }

            inv_jac /= k_r.powi((cartesian_xs.len() - 1) as i32);

            let lo = Into::<T>::into(settings.parameterization.input_rescaling[0][0].0);
            let hi = Into::<T>::into(settings.parameterization.input_rescaling[0][0].1);
            xs[0] = (xs[0] - Into::<T>::into(lo)) / Into::<T>::into(hi - lo);
            inv_jac /= Into::<T>::into(hi - lo);

            let lo = Into::<T>::into(settings.parameterization.input_rescaling[0][1].0);
            let hi = Into::<T>::into(settings.parameterization.input_rescaling[0][1].1);
            xs[1] = (xs[1] - Into::<T>::into(lo)) / Into::<T>::into(hi - lo);
            inv_jac /= Into::<T>::into(hi - lo);

            let lo = Into::<T>::into(settings.parameterization.input_rescaling[0][2].0);
            let hi = Into::<T>::into(settings.parameterization.input_rescaling[0][2].1);
            for x in &mut xs[2..] {
                *x -= Into::<T>::into(lo) / Into::<T>::into(hi - lo);
                inv_jac /= Into::<T>::into(hi - lo);
            }
            (xs, inv_jac)
        }
        _ => {
            if force_radius {
                panic!("Cannot force radius for non-hyper-spherical parameterization.");
            }
            let mut inv_jac = T::one();
            let mut xs = Vec::with_capacity(moms.len() * 3);
            for (i, mom) in moms.iter().enumerate() {
                let (xs_i, inv_jac_i) = inv_parametrize3d(mom, e_cm_squared, i, settings);
                xs.extend(xs_i);
                inv_jac *= inv_jac_i;
            }
            (xs, inv_jac)
        }
    }
}

/// Map a vector in the unit hypercube to the infinite hypercube.
/// Also compute the Jacobian.
pub fn parameterize3d<T: FloatLike>(
    x: &[T],
    e_cm_squared: T,
    loop_index: usize,
    settings: &Settings,
) -> ([T; 3], T) {
    let e_cm =
        e_cm_squared.sqrt() * Into::<T>::into(settings.parameterization.shifts[loop_index].0);
    let mut l_space = [T::zero(); 3];
    let mut jac = T::one();

    // rescale the input to the desired range
    let mut x_r = [T::zero(); 3];
    for (xd, xi, &(lo, hi)) in izip!(
        &mut x_r,
        x,
        &settings.parameterization.input_rescaling[loop_index]
    ) {
        let lo = Into::<T>::into(lo);
        let hi = Into::<T>::into(hi);
        *xd = lo + *xi * (hi - lo);
        jac *= Into::<T>::into(hi - lo);
    }

    match settings.parameterization.mode {
        ParameterizationMode::Cartesian => match settings.parameterization.mapping {
            ParameterizationMapping::Log => {
                for i in 0..3 {
                    let x = x_r[i];
                    l_space[i] = e_cm * (x / (T::one() - x)).ln();
                    jac *= e_cm / (x - x * x);
                }
            }
            ParameterizationMapping::Linear => {
                for i in 0..3 {
                    let x = x_r[i];
                    l_space[i] = e_cm * (T::one() / (T::one() - x) - T::one() / x);
                    jac *=
                        e_cm * (T::one() / (x * x) + T::one() / ((T::one() - x) * (T::one() - x)));
                }
            }
        },
        ParameterizationMode::Spherical => {
            let radius = match settings.parameterization.mapping {
                ParameterizationMapping::Log => {
                    // r = e_cm * ln(1 + b*x/(1-x))
                    let x = x_r[0];
                    let b = Into::<T>::into(settings.parameterization.b);
                    let radius = e_cm * (T::one() + b * x / (T::one() - x)).ln();
                    jac *= e_cm * b / (T::one() - x) / (T::one() + x * (b - T::one()));

                    radius
                }
                ParameterizationMapping::Linear => {
                    // r = e_cm * b * x/(1-x)
                    let b = Into::<T>::into(settings.parameterization.b);
                    let radius = e_cm * b * x_r[0] / (T::one() - x_r[0]);
                    jac *= <T as num_traits::Float>::powi(e_cm * b + radius, 2) / e_cm / b;
                    radius
                }
            };
            let phi = Into::<T>::into(2.) * <T as FloatConst>::PI() * x_r[1];
            jac *= Into::<T>::into(2.) * <T as FloatConst>::PI();

            let cos_theta = -T::one() + Into::<T>::into(2.) * x_r[2]; // out of range
            jac *= Into::<T>::into(2.);
            let sin_theta = (T::one() - cos_theta * cos_theta).sqrt();

            l_space[0] = radius * sin_theta * phi.cos();
            l_space[1] = radius * sin_theta * phi.sin();
            l_space[2] = radius * cos_theta;

            jac *= radius * radius; // spherical coord
        }
        _ => {
            panic!(
                "Inappropriate parameterization mapping specified for parameterize: {:?}.",
                settings.parameterization.mode.clone()
            );
        }
    }

    // add a shift such that k=l is harder to be picked up by integrators such as cuhre
    l_space[0] += e_cm * Into::<T>::into(settings.parameterization.shifts[loop_index].1);
    l_space[1] += e_cm * Into::<T>::into(settings.parameterization.shifts[loop_index].2);
    l_space[2] += e_cm * Into::<T>::into(settings.parameterization.shifts[loop_index].3);

    (l_space, jac)
}

pub fn inv_parametrize3d<T: FloatLike>(
    mom: &LorentzVector<T>,
    e_cm_squared: T,
    loop_index: usize,
    settings: &Settings,
) -> ([T; 3], T) {
    if settings.parameterization.mode != ParameterizationMode::Spherical {
        panic!("Inverse mapping is only implemented for spherical coordinates");
    }

    let mut jac = T::one();
    let e_cm =
        e_cm_squared.sqrt() * Into::<T>::into(settings.parameterization.shifts[loop_index].0);

    let x: T = mom.x - e_cm * Into::<T>::into(settings.parameterization.shifts[loop_index].1);
    let y: T = mom.y - e_cm * Into::<T>::into(settings.parameterization.shifts[loop_index].2);
    let z: T = mom.z - e_cm * Into::<T>::into(settings.parameterization.shifts[loop_index].3);

    let k_r_sq = x * x + y * y + z * z;
    let k_r = k_r_sq.sqrt();

    let x2 = if y < T::zero() {
        T::one() + Into::<T>::into(0.5) * T::FRAC_1_PI() * T::atan2(y, x)
    } else {
        Into::<T>::into(0.5) * T::FRAC_1_PI() * T::atan2(y, x)
    };

    // cover the degenerate case
    if k_r_sq.is_zero() {
        return ([T::zero(), x2, T::zero()], T::zero());
    }

    let x1 = match settings.parameterization.mapping {
        ParameterizationMapping::Log => {
            let b = Into::<T>::into(settings.parameterization.b);
            let x1 = T::one() - b / (-T::one() + b + (k_r / e_cm).exp());
            jac /= e_cm * b / (T::one() - x1) / (T::one() + x1 * (b - T::one()));
            x1
        }
        ParameterizationMapping::Linear => {
            let b = Into::<T>::into(settings.parameterization.b);
            jac /= <T as num_traits::Float>::powi(e_cm * b + k_r, 2) / e_cm / b;
            k_r / (e_cm * b + k_r)
        }
    };

    let x3 = Into::<T>::into(0.5) * (T::one() + z / k_r);

    jac /= Into::<T>::into(2.) * <T as FloatConst>::PI();
    jac /= Into::<T>::into(2.);
    jac /= k_r * k_r;

    let mut x = [x1, x2, x3];
    for (xi, &(lo, hi)) in x
        .iter_mut()
        .zip_eq(&settings.parameterization.input_rescaling[loop_index])
    {
        *xi = (*xi - Into::<T>::into(lo)) / Into::<T>::into(hi - lo);
        jac /= Into::<T>::into(hi - lo);
    }

    (x, jac)
}

pub const MINUTE: usize = 60;
pub const HOUR: usize = 3_600;
pub const DAY: usize = 86_400;
pub const WEEK: usize = 604_800;
pub fn format_wdhms(seconds: usize) -> String {
    let mut compound_duration = vec![];
    if seconds == 0 {
        compound_duration.push(format!("{}", "0s"));
        return compound_duration.join(" ");
    }

    let mut sec = seconds % WEEK;
    // weeks
    let ws = seconds / WEEK;
    if ws != 0 {
        compound_duration.push(format!("{ws}w"));
    }

    // days
    let ds = sec / DAY;
    sec %= DAY;
    if ds != 0 {
        compound_duration.push(format!("{ds}d"));
    }

    // hours
    let hs = sec / HOUR;
    sec %= HOUR;
    if hs != 0 {
        compound_duration.push(format!("{hs}h"));
    }

    // minutes
    let ms = sec / MINUTE;
    sec %= MINUTE;
    if ms != 0 {
        compound_duration.push(format!("{ms}m"));
    }

    // seconds
    if sec != 0 {
        compound_duration.push(format!("{sec}s"));
    }

    compound_duration.join(" ")
}
