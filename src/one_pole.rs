use super::*;

#[inline]
fn g1<const N: usize>(w_c: VFloat<N>) -> VFloat<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let g = g(w_c).abs();
    g / (Simd::splat(1.) + g)
}

/// Contains parameters for an analogue one-pole filter 
pub struct OnePoleParamsSmoothed<const N: usize = FLOATS_PER_VECTOR>
where
    LaneCount<N>: SupportedLaneCount,
{
    g1: LogSmoother<N>,
    k: LogSmoother<N>,
}

impl<const N: usize> OnePoleParamsSmoothed<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    pub fn get_g1(&self) -> &VFloat<N> {
        &self.g1.value
    }

    #[inline]
    pub fn get_gain(&self) -> &VFloat<N> {
        &self.k.value
    }

    #[inline]
    fn set_values(&mut self, g: VFloat<N>, k: VFloat<N>) {
        self.g1.set_all_vals_instantly(g1(g));
        self.k.set_all_vals_instantly(k);
    }

    /// Call this _only_ if you intend to
    /// output non-shelving filter shapes.
    #[inline]
    pub fn set_params(&mut self, w_c: VFloat<N>, gain: VFloat<N>) {
        self.set_values(g(w_c), gain)
    }

    /// Call this _only_ if you intend to output low-shelving filter shapes.
    #[inline]
    pub fn set_params_low_shelving(&mut self, w_c: VFloat<N>, gain: VFloat<N>) {
        self.k.set_all_vals_instantly(gain);
        self.g1.set_all_vals_instantly(g(w_c) / gain.sqrt());
    }

    /// Call this _only_ if you intend to output high-shelving filter shapes.
    #[inline]
    pub fn set_params_high_shelving(&mut self, w_c: VFloat<N>, gain: VFloat<N>) {
        self.k.set_all_vals_instantly(gain);
        self.g1.set_all_vals_instantly(g(w_c) * gain.sqrt());
    }

    #[inline]
    fn set_values_smoothed(&mut self, g: VFloat<N>, k: VFloat<N>, inc: VFloat<N>) {
        self.g1.set_target(g1(g), inc);
        self.k.set_target(k, inc);
    }

    /// Like `Self::set_params` but smoothed
    #[inline]
    pub fn set_params_smoothed(&mut self, w_c: VFloat<N>, gain: VFloat<N>, inc: VFloat<N>) {
        self.set_values_smoothed(g(w_c), gain, inc)
    }

    /// Like `Self::set_params_low_shelving` but smoothed
    #[inline]
    pub fn set_params_low_shelving_smoothed(
        &mut self,
        w_c: VFloat<N>,
        gain: VFloat<N>,
        inc: VFloat<N>,
    ) {
        self.set_values_smoothed(g(w_c) / gain.sqrt(), gain, inc)
    }

    /// Like `Self::set_params_high_shelving` but smoothed.
    #[inline]
    pub fn set_params_high_shelving_smoothed(
        &mut self,
        w_c: VFloat<N>,
        gain: VFloat<N>,
        inc: VFloat<N>,
    ) {
        self.set_values_smoothed(g(w_c) * gain.sqrt(), gain, inc)
    }

    /// Update the filter's internal parameter smoothers.
    ///
    /// After calling `Self::set_params_smoothed([values, ...], num_samples)` this should
    /// be called only _once_ per sample, _up to_ `num_samples` times, until
    /// `Self::set_params_smoothed` is to be called again
    #[inline]
    pub fn update_smoothers(&mut self) {
        self.g1.tick1();
        self.k.tick1();
    }
}

#[derive(Default)]
pub struct OnePole<const N: usize = FLOATS_PER_VECTOR>
where
    LaneCount<N>: SupportedLaneCount,
{
    lp: Integrator<N>,
    x: VFloat<N>,
}

impl<const N: usize> OnePole<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    pub fn reset(&mut self) {
        self.lp.reset()
    }

    /// The "`tick`" method, must be called _only once_ per sample, _every sample_.
    /// 
    /// Feeds `x` into the filter, which updates it's internal state accordingly.
    ///
    /// After calling this, you can get different filter outputs
    /// using `Self::get_{highpass, lowpass, allpass, ...}`
    #[inline]
    pub fn process(&mut self, x: VFloat<N>, g1: VFloat<N>) {

        self.x = x;
        self.lp.process((x - self.lp.state()) * g1);
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
    pub fn get_allpass(&self) -> VFloat<N> {
        self.get_lowpass() - self.get_highpass()
    }

    #[inline]
    pub fn get_highpass(&self) -> VFloat<N> {
        &self.x - self.get_lowpass()
    }

    #[inline]
    pub fn get_low_shelf(&self, gain: VFloat<N>) -> VFloat<N> {
        gain.mul_add(*self.get_lowpass(), self.get_highpass())
    }

    #[inline]
    pub fn get_high_shelf(&self, gain: VFloat<N>) -> VFloat<N> {
        gain.mul_add(self.get_highpass(), *self.get_lowpass())
    }
}

#[cfg(feature = "num")]
pub mod transfer {

    use super::*;

    fn h_denominator<T: Float>(s: Complex<T>) -> Complex<T> {
        s + T::one()
    }

    pub fn low_pass<T: Float>(s: Complex<T>) -> Complex<T> {
        h_denominator(s).finv()
    }

    pub fn all_pass<T: Float>(s: Complex<T>) -> Complex<T> {
        (-s + T::one()).fdiv(h_denominator(s))
    }

    pub fn high_pass<T: Float>(s: Complex<T>) -> Complex<T> {
        s.fdiv(h_denominator(s))
    }

    pub fn low_shelf<T: Float>(s: Complex<T>, gain: T) -> Complex<T> {
        tilting(s, gain.recip()).scale(gain.sqrt())
    }

    pub fn tilting<T: Float>(s: Complex<T>, gain: T) -> Complex<T> {
        let m = gain.sqrt();
        (s.scale(m) + T::one()) / (s + m)
    }

    pub fn high_shelf<T: Float>(s: Complex<T>, gain: T) -> Complex<T> {
        tilting(s, gain).scale(gain.sqrt())
    }
}
