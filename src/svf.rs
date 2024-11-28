use super::*;

/// Smoothers for parameters of an SVF filter
pub struct SVFParamsSmoothed<const N: usize = FLOATS_PER_VECTOR>
where
    LaneCount<N>: SupportedLaneCount,
{
    g: LogSmoother<N>,
    r: LogSmoother<N>,
    k: LogSmoother<N>,
}

impl<const N: usize> SVFParamsSmoothed<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    pub fn get_g(&self) -> &VFloat<N> {
        &self.g.value
    }

    #[inline]
    pub fn get_res(&self) -> &VFloat<N> {
        &self.r.value
    }

    #[inline]
    pub fn get_root_gain(&self) -> &VFloat<N> {
        &self.k.value
    }

    #[inline]
    fn set_values(&mut self, g: VFloat<N>, res: VFloat<N>, gain: VFloat<N>) {
        self.k.set_all_vals_instantly(gain);
        self.g.set_all_vals_instantly(g);
        self.r.set_all_vals_instantly(res);
    }

    /// call this if you intend to use _only_ the low-shelving output
    #[inline]
    pub fn set_params_low_shelving(&mut self, w_c: VFloat<N>, res: VFloat<N>, gain: VFloat<N>) {
        let m2 = gain.sqrt();
        self.set_values(g(w_c) / m2.sqrt(), res, m2);
    }

    /// call this if you intend to use _only_ the band-shelving output
    #[inline]
    pub fn set_params_band_shelving(&mut self, w_c: VFloat<N>, res: VFloat<N>, gain: VFloat<N>) {
        self.set_values(g(w_c), res / gain.sqrt(), gain);
    }

    /// call this if you intend to use _only_ the high-shelving output
    #[inline]
    pub fn set_params_high_shelving(&mut self, w_c: VFloat<N>, res: VFloat<N>, gain: VFloat<N>) {
        let m2 = gain.sqrt();
        self.set_values(g(w_c) * m2.sqrt(), res, m2);
    }

    /// call this if you do not intend to use the shelving outputs
    #[inline]
    pub fn set_params(&mut self, w_c: VFloat<N>, res: VFloat<N>, gain: VFloat<N>) {
        self.set_values(g(w_c), res, gain);
    }

    #[inline]
    fn set_values_smoothed(
        &mut self,
        g: VFloat<N>,
        res: VFloat<N>,
        gain: VFloat<N>,
        inc: VFloat<N>,
    ) {
        self.k.set_target(gain, inc);
        self.g.set_target(g, inc);
        self.r.set_target(res, inc);
    }

    /// Like `Self::set_params_low_shelving` but with smoothing
    #[inline]
    pub fn set_params_low_shelving_smoothed(
        &mut self,
        w_c: VFloat<N>,
        res: VFloat<N>,
        gain: VFloat<N>,
        inc: VFloat<N>,
    ) {
        let m2 = gain.sqrt();
        self.set_values_smoothed(g(w_c) / m2.sqrt(), res, m2, inc);
    }

    /// Like `Self::set_params_band_shelving` but with smoothing
    #[inline]
    pub fn set_params_band_shelving_smoothed(
        &mut self,
        w_c: VFloat<N>,
        res: VFloat<N>,
        gain: VFloat<N>,
        inc: VFloat<N>,
    ) {
        self.set_values_smoothed(g(w_c), res / gain.sqrt(), gain, inc);
    }

    /// Like `Self::set_params_high_shelving` but with smoothing
    #[inline]
    pub fn set_params_high_shelving_smoothed(
        &mut self,
        w_c: VFloat<N>,
        res: VFloat<N>,
        gain: VFloat<N>,
        inc: VFloat<N>,
    ) {
        let m2 = gain.sqrt();
        self.set_values_smoothed(g(w_c) * m2.sqrt(), res, m2, inc);
    }

    /// Like `Self::set_params_non_shelving` but with smoothing
    #[inline]
    pub fn set_params_smoothed(
        &mut self,
        w_c: VFloat<N>,
        res: VFloat<N>,
        gain: VFloat<N>,
        inc: VFloat<N>,
    ) {
        self.g.set_target(g(w_c), inc);
        self.r.set_target(res, inc);
        self.k.set_all_vals_instantly(gain);
    }

    /// Update the filter's internal parameter smoothers.
    ///
    /// After calling `Self::set_params_<output_type>_smoothed(values, ..., num_samples)` this
    /// function should be called _up to_ `num_samples` times, until, that function is to be
    /// called again, calling this function more than `num_samples` times might result in
    /// the internal parameter states diverging away from the previously set values
    #[inline]
    pub fn tick_all_smoothers(&mut self) {
        self.k.tick1();
        self.r.tick1();
        self.g.tick1();
    }
}

/// Digital implementation of the analogue SVF Filter. Based on the
/// one in the book The Art of VA Filter Design by Vadim Zavalishin
///
/// Capable of outputing many different filter types,
/// (highpass, lowpass, bandpass, notch, shelving....)
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
    #[inline]
    pub fn process(
        &mut self,
        x: VFloat<N>,
        g: VFloat<N>,
        res: VFloat<N>,
    ) {
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
    pub fn get_lowpass(
        &self,
    ) -> &VFloat<N> {
        self.lp.output()
    }

    #[inline]
    pub fn get_bandpass(
        &self,
    ) -> &VFloat<N> {
        self.bp.output()
    }

    #[inline]
    pub fn get_unit_bandpass(
        &self,
    ) -> &VFloat<N> {
        &self.bp1
    }

    #[inline]
    pub fn get_highpass(
        &self,
    ) -> &VFloat<N> {
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
    pub fn get_high_shelf(
        &self,
        root_gain: VFloat<N>,
    ) -> VFloat<N> {
        let &hp = self.get_highpass();
        let &bp1 = self.get_unit_bandpass();
        let &lp = self.get_lowpass();
        root_gain.mul_add(root_gain.mul_add(hp, bp1), lp)
    }

    #[inline]
    pub fn get_band_shelf(
        &self,
        root_gain: VFloat<N>,
    ) -> VFloat<N> {
        let &bp1 = self.get_unit_bandpass();
        let &x = self.get_passthrough();
        bp1.mul_add(root_gain, x - bp1)
    }

    #[inline]
    pub fn get_low_shelf(
        &self,
        root_gain: VFloat<N>,
    ) -> VFloat<N> {
        let &hp = self.get_highpass();
        let &bp1 = self.get_unit_bandpass();
        let &lp = self.get_lowpass();
        root_gain.mul_add(root_gain.mul_add(lp, bp1), hp)
    }
}

#[cfg(feature = "num")]
pub mod impedence {

    use super::*;

    fn two<T: Float>(res: T) -> T {
        res + res
    }

    fn h_denominator<T: Float>(s: Complex<T>, res: T) -> Complex<T> {
        s * (s + two(res)) + T::one()
    }

    pub fn low_pass<T: Float>(s: Complex<T>, res: T) -> Complex<T> {
        h_denominator(s, res).finv()
    }

    pub fn band_pass<T: Float>(s: Complex<T>, res: T) -> Complex<T> {
        s.fdiv(h_denominator(s, res))
    }

    pub fn unit_band_pass<T: Float>(s: Complex<T>, res: T) -> Complex<T> {
        band_pass(s, res).scale(two(res))
    }

    pub fn high_pass<T: Float>(s: Complex<T>, res: T) -> Complex<T> {
        (s * s).fdiv(h_denominator(s, res))
    }

    pub fn all_pass<T: Float>(s: Complex<T>, res: T) -> Complex<T> {
        let bp1 = unit_band_pass(s, res);
        bp1 + bp1 - Complex::one()
    }

    pub fn notch<T: Float>(s: Complex<T>, res: T) -> Complex<T> {
        Complex::<T>::one() - unit_band_pass(s, res)
    }

    pub fn tilting<T: Float>(s: Complex<T>, res: T, gain: T) -> Complex<T> {
        let m2 = gain.sqrt();
        let m = m2.sqrt();
        let sm = s.unscale(m);
        (s * s + sm.scale(two(res)) + m2.recip()).fdiv(h_denominator(sm, res))
    }

    pub fn low_shelf<T: Float>(s: Complex<T>, res: T, gain: T) -> Complex<T> {
        let m2 = gain.sqrt();
        tilting(s, res, gain.recip()).scale(m2)
    }

    pub fn band_shelf<T: Float>(s: Complex<T>, res: T, gain: T) -> Complex<T> {
        let m = gain.sqrt();
        (s * (s + two(res) * m) + T::one()).fdiv(h_denominator(s, res / m))
    }

    pub fn high_shelf<T: Float>(s: Complex<T>, res: T, gain: T) -> Complex<T> {
        let m2 = gain.sqrt();
        tilting(s, res, gain).scale(m2)
    }
}
