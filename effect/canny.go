package effect

import (
	"image"
	"math"

	"github.com/anthonynsimon/bild/clone"
	"github.com/anthonynsimon/bild/parallel"
)

// Canny returns an edge map of the image using the Canny edge detection algorithm.
//
// The pipeline consists of Gaussian smoothing of the image luminance, gradient
// computation with the Sobel operator, non-maximum suppression and finally
// double-threshold hysteresis. The result is a binary edge map with the same
// bounds as the source image: edge pixels are opaque white and all other
// pixels are fully transparent.
//
// Parameter sigma is the standard deviation of the Gaussian smoothing kernel.
// A value of 0 (or less, or NaN) disables smoothing. The kernel radius is
// bounded by the largest image dimension, so arbitrarily large (even infinite)
// sigma values terminate deterministically.
//
// Parameters lowThreshold and highThreshold are the hysteresis thresholds on
// the normalized gradient magnitude range of 0 to 255. Values are clamped to
// that range (NaN maps to 0) and swapped if low is greater than high. Edge
// pixels with magnitude >= highThreshold are strong, pixels with magnitude
// >= lowThreshold are weak. Weak pixels are only kept if they are 8-connected
// to a strong pixel.
//
// Luminance is computed from alpha-premultiplied colors, so fully transparent
// pixels contribute no color and their (invisible) RGB values cannot create
// spurious edges. A transition between opaque and transparent regions is an
// alpha discontinuity and may itself produce an edge.
func Canny(src image.Image, sigma, lowThreshold, highThreshold float64) *image.RGBA {
	bounds := src.Bounds()
	if bounds.Empty() {
		return &image.RGBA{}
	}
	w, h := bounds.Dx(), bounds.Dy()

	sigma = cannySigma(sigma)
	lowThreshold = cannyThreshold(lowThreshold)
	highThreshold = cannyThreshold(highThreshold)
	if lowThreshold > highThreshold {
		lowThreshold, highThreshold = highThreshold, lowThreshold
	}

	lum := cannyLuminance(src, w, h)

	if sigma > 0 {
		cannyGaussianBlur(lum, w, h, sigma)
	}

	mag, dir := cannyGradients(lum, w, h)
	suppressed := cannyNonMaxSuppression(mag, dir, w, h)
	edges := cannyHysteresis(suppressed, w, h, lowThreshold, highThreshold)

	dst := image.NewRGBA(bounds)
	parallel.Line(h, func(start, end int) {
		for y := start; y < end; y++ {
			for x := 0; x < w; x++ {
				if edges[y*w+x] {
					pos := y*dst.Stride + x*4
					dst.Pix[pos+0] = 0xFF
					dst.Pix[pos+1] = 0xFF
					dst.Pix[pos+2] = 0xFF
					dst.Pix[pos+3] = 0xFF
				}
			}
		}
	})

	return dst
}

// cannySigma normalizes the sigma parameter. Negative and non-finite
// non-positive values disable smoothing.
func cannySigma(sigma float64) float64 {
	if math.IsNaN(sigma) || sigma < 0 {
		return 0
	}
	return sigma
}

// cannyThreshold normalizes a hysteresis threshold to the range 0 to 255.
func cannyThreshold(t float64) float64 {
	if math.IsNaN(t) || t < 0 {
		return 0
	}
	if t > 255 {
		return 255
	}
	return t
}

// cannyLuminance returns the per-pixel luminance of the image using the
// weights 0.3R + 0.6G + 0.1B. The source is converted to alpha-premultiplied
// RGBA first, so fully transparent pixels contribute a luminance of 0
// regardless of their underlying color values.
func cannyLuminance(src image.Image, w, h int) []float64 {
	rgba := clone.AsRGBA(src)
	lum := make([]float64, w*h)

	parallel.Line(h, func(start, end int) {
		for y := start; y < end; y++ {
			for x := 0; x < w; x++ {
				pos := y*rgba.Stride + x*4
				lum[y*w+x] = 0.3*float64(rgba.Pix[pos+0]) +
					0.6*float64(rgba.Pix[pos+1]) +
					0.1*float64(rgba.Pix[pos+2])
			}
		}
	})

	return lum
}

// cannyGaussianBlur smooths the luminance in place with a separable Gaussian
// kernel of the given standard deviation. The kernel radius is 3*sigma bounded
// by the largest image dimension. Out-of-range samples clamp to the edge.
func cannyGaussianBlur(lum []float64, w, h int, sigma float64) {
	maxRadius := max(w, h)
	r := 3 * sigma
	if r > float64(maxRadius) {
		r = float64(maxRadius)
	}
	radius := int(math.Ceil(r))
	if radius < 1 {
		radius = 1
	}

	length := 2*radius + 1
	kernel := make([]float64, length)
	sum := 0.0
	for i := 0; i < length; i++ {
		x := float64(i - radius)
		kernel[i] = math.Exp(-(x * x) / (2 * sigma * sigma))
		sum += kernel[i]
	}
	for i := range kernel {
		kernel[i] /= sum
	}

	tmp := make([]float64, w*h)

	// Horizontal pass
	parallel.Line(h, func(start, end int) {
		for y := start; y < end; y++ {
			for x := 0; x < w; x++ {
				acc := 0.0
				for k := 0; k < length; k++ {
					ix := x + k - radius
					if ix < 0 {
						ix = 0
					} else if ix >= w {
						ix = w - 1
					}
					acc += lum[y*w+ix] * kernel[k]
				}
				tmp[y*w+x] = acc
			}
		}
	})

	// Vertical pass
	parallel.Line(h, func(start, end int) {
		for y := start; y < end; y++ {
			for x := 0; x < w; x++ {
				acc := 0.0
				for k := 0; k < length; k++ {
					iy := y + k - radius
					if iy < 0 {
						iy = 0
					} else if iy >= h {
						iy = h - 1
					}
					acc += tmp[iy*w+x] * kernel[k]
				}
				lum[y*w+x] = acc
			}
		}
	})
}

// cannyGradients computes the gradient magnitude and quantized direction for
// each pixel using the Sobel operator. Magnitudes are normalized to the range
// 0 to 255. Directions are quantized to the four sectors 0, 45, 90 and 135
// degrees, encoded as 0 to 3. Out-of-range samples clamp to the edge.
func cannyGradients(lum []float64, w, h int) (mag []float64, dir []uint8) {
	mag = make([]float64, w*h)
	dir = make([]uint8, w*h)

	at := func(x, y int) float64 {
		if x < 0 {
			x = 0
		} else if x >= w {
			x = w - 1
		}
		if y < 0 {
			y = 0
		} else if y >= h {
			y = h - 1
		}
		return lum[y*w+x]
	}

	parallel.Line(h, func(start, end int) {
		for y := start; y < end; y++ {
			for x := 0; x < w; x++ {
				gx := at(x+1, y-1) + 2*at(x+1, y) + at(x+1, y+1) -
					at(x-1, y-1) - 2*at(x-1, y) - at(x-1, y+1)
				gy := at(x-1, y+1) + 2*at(x, y+1) + at(x+1, y+1) -
					at(x-1, y-1) - 2*at(x, y-1) - at(x+1, y-1)

				i := y*w + x
				// The maximum Sobel magnitude is 4*255*sqrt(2); normalize to 0-255.
				mag[i] = math.Hypot(gx, gy) / (4 * math.Sqrt2)

				angle := math.Atan2(gy, gx) * 180 / math.Pi
				if angle < 0 {
					angle += 180
				}
				switch {
				case angle < 22.5 || angle >= 157.5:
					dir[i] = 0
				case angle < 67.5:
					dir[i] = 1
				case angle < 112.5:
					dir[i] = 2
				default:
					dir[i] = 3
				}
			}
		}
	})

	return mag, dir
}

// cannyNonMaxSuppression thins the gradient magnitude map by keeping only the
// pixels whose magnitude is a local maximum along their gradient direction.
// The outermost pixel ring is always suppressed.
func cannyNonMaxSuppression(mag []float64, dir []uint8, w, h int) []float64 {
	result := make([]float64, w*h)

	if w < 3 || h < 3 {
		return result
	}

	parallel.Line(h-2, func(start, end int) {
		for y := start + 1; y < end+1; y++ {
			for x := 1; x < w-1; x++ {
				i := y*w + x
				m := mag[i]

				var n0, n1 float64
				switch dir[i] {
				case 0:
					n0, n1 = mag[i-1], mag[i+1]
				case 1:
					n0, n1 = mag[i-w+1], mag[i+w-1]
				case 2:
					n0, n1 = mag[i-w], mag[i+w]
				default:
					n0, n1 = mag[i-w-1], mag[i+w+1]
				}

				if m >= n0 && m >= n1 {
					result[i] = m
				}
			}
		}
	})

	return result
}

// cannyHysteresis applies the double threshold and connectivity analysis.
// Pixels with magnitude >= high are strong seeds; pixels with magnitude >= low
// are kept only if reachable from a strong pixel through 8-connectivity.
func cannyHysteresis(suppressed []float64, w, h int, low, high float64) []bool {
	edges := make([]bool, w*h)
	var stack []int

	for i, v := range suppressed {
		if v >= high {
			edges[i] = true
			stack = append(stack, i)
		}
	}

	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		x, y := i%w, i/w

		for dy := -1; dy <= 1; dy++ {
			for dx := -1; dx <= 1; dx++ {
				if dx == 0 && dy == 0 {
					continue
				}
				nx, ny := x+dx, y+dy
				if nx < 0 || nx >= w || ny < 0 || ny >= h {
					continue
				}
				ni := ny*w + nx
				if !edges[ni] && suppressed[ni] >= low {
					edges[ni] = true
					stack = append(stack, ni)
				}
			}
		}
	}

	return edges
}
