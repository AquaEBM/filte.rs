use super::*;

/// `w_c` must be in the range [0 ; pi[
/// let `g = tan(w_c/2)`
/// returns `g / (1 + g)`
#[inline]
pub fn g1<const N: usize>(w_c: VFloat<N>) -> VFloat<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let g = math::tan_half_x(w_c);
    g / (Simd::splat(1.) + g)
}

/// Digital implementation of the analogue one-pole filter. Based on the
/// one in the book The Art of VA Filter Design by Vadim Zavalishin.
///
/// Capable of outputing many different shapes,
/// (highpass, lowpass, allpass, shelving....)
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
        self.get_passthrough() - self.get_lowpass()
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

    #[inline]
    fn h_denominator<T: Float>(s: Complex<T>) -> Complex<T> {
        s + T::one()
    }

    #[inline]
    pub fn low_pass<T: Float>(s: Complex<T>) -> Complex<T> {
        h_denominator(s).finv()
    }

    #[inline]
    pub fn all_pass<T: Float>(s: Complex<T>) -> Complex<T> {
        (-s + T::one()).fdiv(h_denominator(s))
    }

    #[inline]
    pub fn high_pass<T: Float>(s: Complex<T>) -> Complex<T> {
        s.fdiv(h_denominator(s))
    }

    #[inline]
    pub fn low_shelf<T: Float>(s: Complex<T>, gain: T) -> Complex<T> {
        tilting(s, gain.recip()).scale(gain.sqrt())
    }

    #[inline]
    pub fn tilting<T: Float>(s: Complex<T>, gain: T) -> Complex<T> {
        let m = gain.sqrt();
        (s.scale(m) + T::one()) / (s + m)
    }

    #[inline]
    pub fn high_shelf<T: Float>(s: Complex<T>, gain: T) -> Complex<T> {
        tilting(s, gain).scale(gain.sqrt())
    }
}
