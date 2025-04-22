use pyo3::prelude::*;


#[inline]
fn squared(v: &Vec<f64>) -> f64 {
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

#[inline]
fn add(v1: &Vec<f64>, v2: &Vec<f64>) -> Vec<f64> {
    return vec![v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]];
}

#[inline]
fn sub(v1: &Vec<f64>, v2: &Vec<f64>) -> Vec<f64> {
    return vec![v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]];
}

#[inline]
fn spatial(v: &Vec<f64>) -> Vec<f64> {
    return vec![v[1], v[2], v[3]];
}

fn propagators(m_psi: f64, k: Vec<f64>, q: Vec<f64>, p: Vec<f64>) -> Vec<f64> {
    let m_squared = m_psi*m_psi;
    let prop0 = squared(&k) + m_squared;
    let prop1 = squared(&sub(&k, &spatial(&q))) + m_squared;
    let prop2 = squared(&add(&k, &spatial(&p))) + m_squared;

    return vec![prop0, prop1, prop2];
}
#[pyfunction]
fn ltd_triangle(m_psi: f64, k: Vec<f64>, q: Vec<f64>, p: Vec<f64>) -> PyResult<f64> {
    // Implement the improved LTD expression here (Ex 2.4)
    let mut res: f64 = 0.;
    let pi_factor = 64.*std::f64::consts::PI.powi(3);
    let q_i= [0., -q[0], p[0]];
    let props = propagators(m_psi, k, q, p);
    let ose = [
        props[0].sqrt(),
        props[1].sqrt(),
        props[2].sqrt()
    ];
    let mut eta = std::collections::HashMap::new();

    for i in 0..3{
        for j in 0..3{
            if i != j{
                eta.insert((i, j), ose[i] + ose[j] - q_i[i] + q_i[j]);
            }
        }
    }
    
    res += 1./eta.get(&(2, 0)).unwrap()/eta.get(&(2, 1)).unwrap();
    res += 1./eta.get(&(0, 1)).unwrap()/eta.get(&(0, 2)).unwrap();
    res += 1./eta.get(&(0, 1)).unwrap()/eta.get(&(2, 1)).unwrap();
    res += 1./eta.get(&(0, 2)).unwrap()/eta.get(&(1, 2)).unwrap();
    res += 1./eta.get(&(1, 0)).unwrap()/eta.get(&(1, 2)).unwrap();
    res += 1./eta.get(&(1, 0)).unwrap()/eta.get(&(2, 0)).unwrap();
    res /= pi_factor;

    Ok(res)
}

#[pyfunction]
fn prop_factor(m_psi: f64, k: Vec<f64>, q: Vec<f64>, p: Vec<f64>, weight: f64) -> PyResult<f64> {
    // This is equivalent to f(k, p, q, m_psi)=1/8
    let props = propagators(m_psi, k, q, p);

    let res: f64 = (props[0]*props[1]*props[2]).powf(-weight);

    Ok(res)
}

#[pyfunction]
fn get_ch_wgt(m_psi: f64, k: Vec<f64>, q: Vec<f64>, p: Vec<f64>, channel: usize, mc_exp: f64) -> PyResult<f64> {
    // Returns the channel weight for multichanneling
    let props = propagators(m_psi, k, q, p);
    let wgt = props[channel].powf(-mc_exp)/(props[0].powf(-mc_exp) + props[1].powf(-mc_exp) + props[2].powf(-mc_exp));

    Ok(wgt)
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn triangle(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ltd_triangle, m)?)?;
    m.add_function(wrap_pyfunction!(prop_factor, m)?)?;
    m.add_function(wrap_pyfunction!(get_ch_wgt, m)?)?;
    Ok(())
}
