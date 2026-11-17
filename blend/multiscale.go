package blend

import (
	"errors"
	"fmt"
	"image"
	"image/color"

	"github.com/anthonynsimon/bild/math/f64"
	"github.com/anthonynsimon/bild/parallel"
)

// reduceKernel is the 5-tap binomial filter applied before decimation when
// building a Gaussian pyramid.
var reduceKernel = [5]float64{1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16}

// expandKernel is twice the reduce kernel. The factor of two compensates the
// energy lost to the zero-inserted samples when upsampling.
var expandKernel = [5]float64{2.0 / 16, 8.0 / 16, 12.0 / 16, 8.0 / 16, 2.0 / 16}

// Multiscale blends the bg and fg images using multi-scale (Laplacian pyramid)
// fusion and returns the resulting image.
//
// Instead of blending the full images with a single weight per pixel — which
// produces a blurry, ghosted seam across wide transitions — each image is
// decomposed into frequency bands and every band is blended with the mask
// low-pass filtered to the matching scale. Low frequencies therefore
// transition smoothly across the whole blend region while high frequency
// detail from both inputs is preserved, and the result is reconstructed at
// the original resolution.
//
// The mask is a grayscale image whose value selects the source per pixel:
// black (0) takes bg, white (255) takes fg, and intermediate values blend
// both. Its bounds must fully cover the bg bounds.
//
// Parameter levels is the number of pyramid levels used for the fusion. It
// must be at least 1 (1 degenerates to a plain per-pixel mask blend) and is
// clamped to the maximum useful depth for the image size. Peak memory grows
// with the number of levels but stays bounded by a small multiple of the
// input size, since each level holds a quarter of the pixels of the previous
// one.
//
// The inputs are sampled through their own bounds, so non-zero Min points and
// differing image types are handled transparently, and they are never
// modified. Alpha is blended in premultiplied form so fully transparent
// regions contribute no color. The result has the same bounds as bg.
func Multiscale(bg, fg image.Image, mask image.Image, levels int) (*image.RGBA, error) {
	if bg == nil || fg == nil {
		return nil, errors.New("multiscale: bg and fg images must not be nil")
	}
	bb := bg.Bounds()
	w, h := bb.Dx(), bb.Dy()
	if w < 1 || h < 1 {
		return nil, fmt.Errorf("multiscale: bg image is empty, bounds %v", bb)
	}
	if fb := fg.Bounds(); fb.Dx() != w || fb.Dy() != h {
		return nil, fmt.Errorf("multiscale: image sizes differ, bg is %dx%d but fg is %dx%d", w, h, fb.Dx(), fb.Dy())
	}
	if levels < 1 {
		return nil, fmt.Errorf("multiscale: levels must be at least 1, got %d", levels)
	}
	if mask == nil {
		return nil, errors.New("multiscale: mask image must not be nil")
	}
	if mb := mask.Bounds(); !bb.In(mb) {
		return nil, fmt.Errorf("multiscale: mask bounds %v do not cover image bounds %v", mb, bb)
	}

	levels = min(levels, maxPyramidLevels(w, h))

	ga := gaussianPyramid(toF64Img(bg), levels)
	gb := gaussianPyramid(toF64Img(fg), levels)
	gm := gaussianPyramid(maskToF64Img(mask, bb), levels)

	// Blend the Laplacian band of each level, reusing the Gaussian slices to
	// keep peak memory bounded. The coarsest level holds no band and its
	// Gaussian is blended directly.
	for i := 0; i < levels-1; i++ {
		la := subF64Img(ga[i], upsample(ga[i+1], ga[i].w, ga[i].h))
		lb := subF64Img(gb[i], upsample(gb[i+1], gb[i].w, gb[i].h))
		ga[i] = weightedBlend(lb, la, gm[i])
		gb[i] = nil
		gm[i] = nil
	}
	ga[levels-1] = weightedBlend(gb[levels-1], ga[levels-1], gm[levels-1])
	gb[levels-1] = nil
	gm[levels-1] = nil

	// Collapse the blended pyramid back to the original resolution.
	cur := ga[levels-1]
	for i := levels - 2; i >= 0; i-- {
		cur = addF64Img(upsample(cur, ga[i].w, ga[i].h), ga[i])
		ga[i] = nil
	}

	return toRGBA(cur, bb), nil
}

// maxPyramidLevels returns the number of levels after which further
// decimation no longer reduces an image of the given size.
func maxPyramidLevels(w, h int) int {
	levels := 1
	for w > 1 || h > 1 {
		w = (w + 1) / 2
		h = (h + 1) / 2
		levels++
	}
	return levels
}

// f64Img is a float64 image with nch interleaved channels in the range 0 to 1.
type f64Img struct {
	w, h, nch int
	pix       []float64
}

func newF64Img(w, h, nch int) *f64Img {
	return &f64Img{w: w, h: h, nch: nch, pix: make([]float64, w*h*nch)}
}

// toF64Img converts src to premultiplied float64 RGBA. The RGBA() color model
// already returns alpha-premultiplied values, so transparent pixels contribute
// no color to the fusion.
func toF64Img(src image.Image) *f64Img {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	dst := newF64Img(w, h, 4)

	parallel.Line(h, func(start, end int) {
		for y := start; y < end; y++ {
			pos := y * w * 4
			for x := 0; x < w; x++ {
				r, g, bl, a := src.At(b.Min.X+x, b.Min.Y+y).RGBA()
				dst.pix[pos+0] = float64(r) / 65535
				dst.pix[pos+1] = float64(g) / 65535
				dst.pix[pos+2] = float64(bl) / 65535
				dst.pix[pos+3] = float64(a) / 65535
				pos += 4
			}
		}
	})

	return dst
}

// maskToF64Img converts the region of mask covered by b to a single channel
// float64 image of blend weights in the range 0 to 1.
func maskToF64Img(mask image.Image, b image.Rectangle) *f64Img {
	w, h := b.Dx(), b.Dy()
	dst := newF64Img(w, h, 1)

	parallel.Line(h, func(start, end int) {
		for y := start; y < end; y++ {
			pos := y * w
			for x := 0; x < w; x++ {
				g := color.GrayModel.Convert(mask.At(b.Min.X+x, b.Min.Y+y)).(color.Gray)
				dst.pix[pos] = float64(g.Y) / 255
				pos++
			}
		}
	})

	return dst
}

// toRGBA converts a premultiplied float64 RGBA image back to 8-bit,
// unpremultiplying the color channels. The result spans the given bounds.
func toRGBA(src *f64Img, bounds image.Rectangle) *image.RGBA {
	dst := image.NewRGBA(bounds)

	parallel.Line(src.h, func(start, end int) {
		for y := start; y < end; y++ {
			spos := y * src.w * 4
			dpos := y * dst.Stride
			for x := 0; x < src.w; x++ {
				a := f64.Clamp(src.pix[spos+3], 0, 1)
				r, g, b := src.pix[spos+0], src.pix[spos+1], src.pix[spos+2]
				if a > 0 {
					r /= a
					g /= a
					b /= a
				}
				dst.Pix[dpos+0] = uint8(f64.Clamp(r, 0, 1)*255 + 0.5)
				dst.Pix[dpos+1] = uint8(f64.Clamp(g, 0, 1)*255 + 0.5)
				dst.Pix[dpos+2] = uint8(f64.Clamp(b, 0, 1)*255 + 0.5)
				dst.Pix[dpos+3] = uint8(a*255 + 0.5)
				spos += 4
				dpos += 4
			}
		}
	})

	return dst
}

// gaussianPyramid returns levels successively blurred and halved versions of
// src, starting with src itself. Odd dimensions round up so no edge samples
// are dropped.
func gaussianPyramid(src *f64Img, levels int) []*f64Img {
	pyr := make([]*f64Img, levels)
	pyr[0] = src
	for i := 1; i < levels; i++ {
		pyr[i] = downsample(pyr[i-1])
	}
	return pyr
}

// downsample returns src convolved with the reduce kernel and decimated by
// two in each dimension. Edges are handled by extending the closest pixel.
func downsample(src *f64Img) *f64Img {
	ow, oh := (src.w+1)/2, (src.h+1)/2

	// Horizontal pass at full height.
	tmp := newF64Img(ow, src.h, src.nch)
	parallel.Line(src.h, func(start, end int) {
		for y := start; y < end; y++ {
			for x := 0; x < ow; x++ {
				dpos := (y*ow + x) * src.nch
				for k := 0; k < 5; k++ {
					ix := clampInt(2*x+k-2, 0, src.w-1)
					spos := (y*src.w + ix) * src.nch
					w := reduceKernel[k]
					for c := 0; c < src.nch; c++ {
						tmp.pix[dpos+c] += w * src.pix[spos+c]
					}
				}
			}
		}
	})

	// Vertical pass at reduced width.
	dst := newF64Img(ow, oh, src.nch)
	parallel.Line(oh, func(start, end int) {
		for y := start; y < end; y++ {
			for x := 0; x < ow; x++ {
				dpos := (y*ow + x) * src.nch
				for k := 0; k < 5; k++ {
					iy := clampInt(2*y+k-2, 0, src.h-1)
					spos := (iy*ow + x) * src.nch
					w := reduceKernel[k]
					for c := 0; c < src.nch; c++ {
						dst.pix[dpos+c] += w * tmp.pix[spos+c]
					}
				}
			}
		}
	})

	return dst
}

// upsample returns src zero-insert upsampled to tw by th and convolved with
// the expand kernel. Edges are handled by extending the closest pixel.
func upsample(src *f64Img, tw, th int) *f64Img {
	// Vertical pass, inserting a zero row between source rows.
	tmp := newF64Img(src.w, th, src.nch)
	parallel.Line(th, func(start, end int) {
		for y := start; y < end; y++ {
			for x := 0; x < src.w; x++ {
				dpos := (y*src.w + x) * src.nch
				for k := 0; k < 5; k++ {
					q := y + k - 2
					if q%2 != 0 {
						continue
					}
					iy := clampInt(q/2, 0, src.h-1)
					spos := (iy*src.w + x) * src.nch
					w := expandKernel[k]
					for c := 0; c < src.nch; c++ {
						tmp.pix[dpos+c] += w * src.pix[spos+c]
					}
				}
			}
		}
	})

	// Horizontal pass, inserting a zero column between source columns.
	dst := newF64Img(tw, th, src.nch)
	parallel.Line(th, func(start, end int) {
		for y := start; y < end; y++ {
			for x := 0; x < tw; x++ {
				dpos := (y*tw + x) * src.nch
				for k := 0; k < 5; k++ {
					q := x + k - 2
					if q%2 != 0 {
						continue
					}
					ix := clampInt(q/2, 0, src.w-1)
					spos := (y*src.w + ix) * src.nch
					w := expandKernel[k]
					for c := 0; c < src.nch; c++ {
						dst.pix[dpos+c] += w * tmp.pix[spos+c]
					}
				}
			}
		}
	})

	return dst
}

// subF64Img returns a - b. Both images must have the same dimensions.
func subF64Img(a, b *f64Img) *f64Img {
	dst := newF64Img(a.w, a.h, a.nch)
	parallel.Line(len(dst.pix), func(start, end int) {
		for i := start; i < end; i++ {
			dst.pix[i] = a.pix[i] - b.pix[i]
		}
	})
	return dst
}

// addF64Img returns a + b. Both images must have the same dimensions.
func addF64Img(a, b *f64Img) *f64Img {
	dst := newF64Img(a.w, a.h, a.nch)
	parallel.Line(len(dst.pix), func(start, end int) {
		for i := start; i < end; i++ {
			dst.pix[i] = a.pix[i] + b.pix[i]
		}
	})
	return dst
}

// weightedBlend returns fg*w + bg*(1-w) per pixel, where w is the single
// channel weight image m. All images must have the same dimensions.
func weightedBlend(fg, bg, m *f64Img) *f64Img {
	dst := newF64Img(fg.w, fg.h, fg.nch)
	parallel.Line(fg.h, func(start, end int) {
		for y := start; y < end; y++ {
			pos := y * fg.w * fg.nch
			mpos := y * fg.w
			for x := 0; x < fg.w; x++ {
				w := m.pix[mpos]
				iw := 1 - w
				for c := 0; c < fg.nch; c++ {
					dst.pix[pos+c] = fg.pix[pos+c]*w + bg.pix[pos+c]*iw
				}
				pos += fg.nch
				mpos++
			}
		}
	})
	return dst
}

func clampInt(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
