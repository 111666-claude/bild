/*Package guidedfilter provides edge-preserving smoothing using a guidance image.*/
package guidedfilter

import (
	"image"
	"math"

	"github.com/anthonynsimon/bild/clone"
	"github.com/anthonynsimon/bild/parallel"
)

// Filter returns an edge-preserving smoothed version of src, guided by the
// structure of the guide image. It implements the guided filter of
// He, Sun and Tang ("Guided Image Filtering", ECCV 2010) using a grayscale
// (luminance) projection of the guide, applied independently to each color
// channel of src. The alpha channel is copied unchanged from src.
//
// The radius parameter sets the half-size of the square averaging window and
// must be larger than 0. The epsilon parameter is a regularization term
// controlling how strongly the output follows the guide's edges: larger
// values smooth more aggressively across weak guide edges.
//
// The guide must have the same dimensions as src. If the sizes differ, the
// radius is not positive, or epsilon is not a finite non-negative number, a
// plain RGBA copy of src is returned. Windows are clipped at the image
// borders, so edge pixels are averaged over the intersecting area only.
// The result keeps the bounds of src and never mutates the inputs.
func Filter(src, guide image.Image, radius int, epsilon float64) *image.RGBA {
	bounds := src.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	if w == 0 || h == 0 || radius <= 0 {
		return clone.AsRGBA(src)
	}

	gb := guide.Bounds()
	if gb.Dx() != w || gb.Dy() != h {
		return clone.AsRGBA(src)
	}

	if math.IsNaN(epsilon) || math.IsInf(epsilon, 0) || epsilon < 0 {
		epsilon = 0
	}

	srcRGBA := clone.AsRGBA(src)
	n := w * h

	// Grayscale (luminance) projection of the guide image.
	g := make([]float64, n)
	parallel.Line(h, func(start, end int) {
		for y := start; y < end; y++ {
			for x := 0; x < w; x++ {
				r, gg, b, _ := guide.At(gb.Min.X+x, gb.Min.Y+y).RGBA()
				g[y*w+x] = (0.299*float64(r) + 0.587*float64(gg) + 0.114*float64(b)) / 65535
			}
		}
	})

	// Integral images of the guide and its square, reused across channels.
	iG := integral(g, w, h)
	g2 := make([]float64, n)
	for i, v := range g {
		g2[i] = v * v
	}
	iG2 := integral(g2, w, h)

	p := make([]float64, n)
	pg := make([]float64, n)
	a := make([]float64, n)
	b := make([]float64, n)

	dst := image.NewRGBA(bounds)

	for ch := 0; ch < 3; ch++ {
		// Extract the channel from the source copy.
		parallel.Line(h, func(start, end int) {
			for y := start; y < end; y++ {
				for x := 0; x < w; x++ {
					pos := y*srcRGBA.Stride + x*4
					p[y*w+x] = float64(srcRGBA.Pix[pos+ch]) / 255
				}
			}
		})

		for i := 0; i < n; i++ {
			pg[i] = p[i] * g[i]
		}
		iP := integral(p, w, h)
		iPG := integral(pg, w, h)

		// Per-pixel linear coefficients a, b of the local model p ~ a*g + b.
		parallel.Line(h, func(start, end int) {
			for y := start; y < end; y++ {
				for x := 0; x < w; x++ {
					i := y*w + x
					meanG, _ := windowMean(iG, w, h, x, y, radius)
					meanP, _ := windowMean(iP, w, h, x, y, radius)
					meanGG, _ := windowMean(iG2, w, h, x, y, radius)
					meanPG, _ := windowMean(iPG, w, h, x, y, radius)

					varG := meanGG - meanG*meanG
					if varG < 0 {
						// Guard against tiny negative values from
						// floating point cancellation.
						varG = 0
					}
					covPG := meanPG - meanG*meanP

					denom := varG + epsilon
					if denom <= 0 || math.IsInf(denom, 0) || math.IsNaN(denom) {
						// Zero-variance (flat) guide region with no
						// regularization: fall back to the local mean.
						a[i] = 0
					} else {
						a[i] = covPG / denom
					}
					b[i] = meanP - a[i]*meanG
				}
			}
		})

		// Average the coefficients over the window and compose the output.
		iA := integral(a, w, h)
		iB := integral(b, w, h)
		parallel.Line(h, func(start, end int) {
			for y := start; y < end; y++ {
				for x := 0; x < w; x++ {
					i := y*w + x
					meanA, _ := windowMean(iA, w, h, x, y, radius)
					meanB, _ := windowMean(iB, w, h, x, y, radius)

					q := meanA*g[i] + meanB
					if q < 0 {
						q = 0
					} else if q > 1 {
						q = 1
					}

					pos := y*dst.Stride + x*4
					dst.Pix[pos+ch] = uint8(q*255 + 0.5)
				}
			}
		})
	}

	// Copy the alpha channel unchanged from the source.
	parallel.Line(h, func(start, end int) {
		for y := start; y < end; y++ {
			for x := 0; x < w; x++ {
				dst.Pix[y*dst.Stride+x*4+3] = srcRGBA.Pix[y*srcRGBA.Stride+x*4+3]
			}
		}
	})

	return dst
}

// integral returns the summed-area table of src with size (w+1) x (h+1).
func integral(src []float64, w, h int) []float64 {
	stride := w + 1
	ii := make([]float64, (w+1)*(h+1))
	for y := 0; y < h; y++ {
		var rowsum float64
		for x := 0; x < w; x++ {
			rowsum += src[y*w+x]
			ii[(y+1)*stride+x+1] = ii[y*stride+x+1] + rowsum
		}
	}
	return ii
}

// windowMean returns the mean of the values in the square window of half-size
// r centered at (x, y), clipped to the image bounds, along with the number of
// pixels in the clipped window. ii must be the summed-area table of the
// values, as returned by integral.
func windowMean(ii []float64, w, h, x, y, r int) (mean, count float64) {
	x0 := x - r
	if x0 < 0 {
		x0 = 0
	}
	y0 := y - r
	if y0 < 0 {
		y0 = 0
	}
	x1 := x + r + 1
	if x1 > w {
		x1 = w
	}
	y1 := y + r + 1
	if y1 > h {
		y1 = h
	}

	stride := w + 1
	sum := ii[y1*stride+x1] - ii[y0*stride+x1] - ii[y1*stride+x0] + ii[y0*stride+x0]
	count = float64((x1 - x0) * (y1 - y0))

	return sum / count, count
}
