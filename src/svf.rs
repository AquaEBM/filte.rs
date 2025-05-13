use super::*;

/// Digital implementation of the analogue SVF Filter. Based on the
/// one in the book The Art of VA Filter Design by Vadim Zavalishin
///
/// Capable of outputing many different shapes,
/// (highpass, lowpass, bandpass, allpass, notch, shelving....)
#[derive(Default)]
pub struct SVF<const N: usize = FLOATS_PER_VECTOR>
where
    LaneCount<N>: SupportedLaneCount,
{
    x: VFloat<N>,
    hp: VFloat<N>,
    bp: Integrator<N>,
    bp1: VFloat<N>,
    lp: Integrator<N>,
}

impl<const N: usize> SVF<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    pub fn reset(&mut self) {
        for i in [&mut self.bp, &mut self.lp] {
            i.reset();
        }
    }

    /// Update the filter's internal state.
    ///
    /// This should be called _only once_ per sample, _every sample_
    ///
    /// After calling this, you can get different filter outputs
    /// using `Self::get_{highpass, bandpass, notch, ...}`
    ///
    /// `x` is the input sample fed to the filter.
    ///
    /// `g` is the integrator pre-gain: If `0 <= w_c < pi` is the cutoff frequency,
    /// in radians per sample, then, `g = tan(w_c/2)`.
    ///
    /// `res` is the resonance value of the filter. `0 <= res < 2` must hold.
    /// Values outside of that range may result in instability.
    #[inline]
    pub fn process(&mut self, x: VFloat<N>, g: VFloat<N>, res: VFloat<N>) {
        self.x = x;
        let &bp_s = self.bp.state();
        let &lp_s = self.lp.state();

        let g1 = res + g;

        self.hp = g1.mul_add(-bp_s, self.x - lp_s) / g1.mul_add(g, Simd::splat(1.));

        self.bp.process(self.hp * g);
        let &bp = self.bp.output();
        self.bp1 = bp * res;
        self.lp.process(bp * g);
    }

    #[inline]
    pub fn get_passthrough(&self) -> &VFloat<N> {
        &self.x
    }

    #[inline]
    pub fn get_lowpass(&self) -> &VFloat<N> {
        self.lp.output()
    }

    #[inline]
    pub fn get_bandpass(&self) -> &VFloat<N> {
        self.bp.output()
    }

    #[inline]
    pub fn get_unit_bandpass(&self) -> &VFloat<N> {
        &self.bp1
    }

    #[inline]
    pub fn get_highpass(&self) -> &VFloat<N> {
        &self.hp
    }

    #[inline]
    pub fn get_allpass(&self) -> VFloat<N> {
        // 2 * bp1 - x
        self.get_unit_bandpass().mul_add(Simd::splat(2.), -self.x)
    }

    #[inline]
    pub fn get_notch(&self) -> VFloat<N> {
        // x - bp1
        self.get_passthrough() - self.get_unit_bandpass()
    }

    #[inline]
    pub fn get_high_shelf(&self, root_gain: VFloat<N>) -> VFloat<N> {
        let &hp = self.get_highpass();
        let &bp1 = self.get_unit_bandpass();
        let &lp = self.get_lowpass();
        root_gain.mul_add(root_gain.mul_add(hp, bp1), lp)
    }

    #[inline]
    pub fn get_band_shelf(&self, root_gain: VFloat<N>) -> VFloat<N> {
        let &bp1 = self.get_unit_bandpass();
        let &x = self.get_passthrough();
        bp1.mul_add(root_gain, x - bp1)
    }

    #[inline]
    pub fn get_low_shelf(&self, root_gain: VFloat<N>) -> VFloat<N> {
        let &hp = self.get_highpass();
        let &bp1 = self.get_unit_bandpass();
        let &lp = self.get_lowpass();
        root_gain.mul_add(root_gain.mul_add(lp, bp1), hp)
    }
}

#[cfg(feature = "num")]
pub mod trnasfer {

    use super::*;

    #[inline]
    fn two<T: Float>(res: T) -> T {
        res + res
    }

    #[inline]
    fn h_denominator<T: Float>(s: Complex<T>, res: T) -> Complex<T> {
        s * (s + two(res)) + T::one()
    }

    #[inline]
    pub fn low_pass<T: Float>(s: Complex<T>, res: T) -> Complex<T> {
        h_denominator(s, res).finv()
    }

    #[inline]
    pub fn band_pass<T: Float>(s: Complex<T>, res: T) -> Complex<T> {
        s.fdiv(h_denominator(s, res))
    }

    #[inline]
    pub fn unit_band_pass<T: Float>(s: Complex<T>, res: T) -> Complex<T> {
        band_pass(s, res).scale(two(res))
    }

    #[inline]
    pub fn high_pass<T: Float>(s: Complex<T>, res: T) -> Complex<T> {
        (s * s).fdiv(h_denominator(s, res))
    }

    #[inline]
    pub fn all_pass<T: Float>(s: Complex<T>, res: T) -> Complex<T> {
        let bp1 = unit_band_pass(s, res);
        bp1 + bp1 - Complex::one()
    }

    #[inline]
    pub fn notch<T: Float>(s: Complex<T>, res: T) -> Complex<T> {
        Complex::<T>::one() - unit_band_pass(s, res)
    }

    #[inline]
    pub fn tilting<T: Float>(s: Complex<T>, res: T, gain: T) -> Complex<T> {
        let m2 = gain.sqrt();
        let m = m2.sqrt();
        let sm = s.unscale(m);
        (s * s + sm.scale(two(res)) + m2.recip()).fdiv(h_denominator(sm, res))
    }

    #[inline]
    pub fn low_shelf<T: Float>(s: Complex<T>, res: T, gain: T) -> Complex<T> {
        let m2 = gain.sqrt();
        tilting(s, res, gain.recip()).scale(m2)
    }

    #[inline]
    pub fn band_shelf<T: Float>(s: Complex<T>, res: T, gain: T) -> Complex<T> {
        let m = gain.sqrt();
        (s * (s + two(res) * m) + T::one()).fdiv(h_denominator(s, res / m))
    }

    #[inline]
    pub fn high_shelf<T: Float>(s: Complex<T>, res: T, gain: T) -> Complex<T> {
        let m2 = gain.sqrt();
        tilting(s, res, gain).scale(m2)
    }
}
