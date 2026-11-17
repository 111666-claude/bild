package transform

import (
	"image"
	"math"
	"reflect"
	"testing"

	"github.com/anthonynsimon/bild/util"
)

// decodeTestNormal decodes the RGB channels of a tangent-space normal map into a vector.
func decodeTestNormal(r, g, b uint8) (float64, float64, float64) {
	return float64(r)*(2.0/255.0) - 1, float64(g)*(2.0/255.0) - 1, float64(b)*(2.0/255.0) - 1
}

// encodeTestNormal encodes a vector the same way tangent-space normal maps store it.
func encodeTestNormal(x, y, z float64) (uint8, uint8, uint8) {
	enc := func(v float64) uint8 {
		return uint8(math.Round((v + 1) * 127.5))
	}
	return enc(x), enc(y), enc(z)
}

// normalAtPix returns the decoded, unit-length normal at the given pixel.
func normalAtPix(img *image.RGBA, x, y int) (float64, float64, float64) {
	pos := y*img.Stride + x*4
	nx, ny, nz := decodeTestNormal(img.Pix[pos], img.Pix[pos+1], img.Pix[pos+2])
	length := math.Hypot(math.Hypot(nx, ny), nz)
	return nx / length, ny / length, nz / length
}

// assertUnitNormal checks that the pixel's vector is unit length and close to the expected
// direction.
func assertUnitNormal(t *testing.T, name string, img *image.RGBA, x, y int, ex, ey, ez, tolerance float64) {
	t.Helper()
	pos := y*img.Stride + x*4
	nx, ny, nz := decodeTestNormal(img.Pix[pos], img.Pix[pos+1], img.Pix[pos+2])
	length := math.Hypot(math.Hypot(nx, ny), nz)
	if math.Abs(length-1) > tolerance {
		t.Errorf("%s: pixel (%d,%d): expected unit length, got %f (rgb: %d,%d,%d)",
			name, x, y, length, img.Pix[pos], img.Pix[pos+1], img.Pix[pos+2])
	}
	if d := math.Abs(nx - ex); d > tolerance {
		t.Errorf("%s: pixel (%d,%d): x component expected %f, got %f", name, x, y, ex, nx)
	}
	if d := math.Abs(ny - ey); d > tolerance {
		t.Errorf("%s: pixel (%d,%d): y component expected %f, got %f", name, x, y, ey, ny)
	}
	if d := math.Abs(nz - ez); d > tolerance {
		t.Errorf("%s: pixel (%d,%d): z component expected %f, got %f", name, x, y, ez, nz)
	}
}

func TestResizeNormalMapConstantDirection(t *testing.T) {
	// A constant analytic normal field: every pixel points along normalize(3, 0, 4).
	dirX, dirY, dirZ := 0.6, 0.0, 0.8
	r, g, b := encodeTestNormal(dirX, dirY, dirZ)

	src := image.NewRGBA(image.Rect(0, 0, 4, 3))
	for y := 0; y < 3; y++ {
		for x := 0; x < 4; x++ {
			pos := y*src.Stride + x*4
			src.Pix[pos+0] = r
			src.Pix[pos+1] = g
			src.Pix[pos+2] = b
			src.Pix[pos+3] = 0xFF
		}
	}

	// Use the decoded (quantized) direction as the reference.
	exX, exY, exZ := decodeTestNormal(r, g, b)
	length := math.Hypot(math.Hypot(exX, exY), exZ)
	exX, exY, exZ = exX/length, exY/length, exZ/length

	filters := []struct {
		name   string
		filter ResampleFilter
	}{
		{"NearestNeighbor", NearestNeighbor},
		{"Box", Box},
		{"Linear", Linear},
		{"Gaussian", Gaussian},
		{"MitchellNetravali", MitchellNetravali},
		{"CatmullRom", CatmullRom},
		{"Lanczos", Lanczos},
	}

	sizes := []struct {
		width, height int
	}{
		{1, 1},
		{2, 2},
		{8, 6},
		{3, 5},
	}

	for _, f := range filters {
		for _, s := range sizes {
			name := f.name
			actual := ResizeNormalMap(src, s.width, s.height, f.filter)
			if actual.Bounds() != image.Rect(0, 0, s.width, s.height) {
				t.Fatalf("%s: expected bounds %v, got %v", name, image.Rect(0, 0, s.width, s.height), actual.Bounds())
			}
			for y := 0; y < s.height; y++ {
				for x := 0; x < s.width; x++ {
					assertUnitNormal(t, name, actual, x, y, exX, exY, exZ, 0.01)
					if alpha := actual.Pix[y*actual.Stride+x*4+3]; alpha != 0xFF {
						t.Errorf("%s: pixel (%d,%d): expected alpha 255, got %d", name, x, y, alpha)
					}
				}
			}
		}
	}
}

func TestResizeNormalMapDirectionInterpolation(t *testing.T) {
	// Horizontal field of two analytic normals, mirrored around the z axis.
	left := [3]float64{-0.6, 0, 0.8}
	right := [3]float64{0.6, 0, 0.8}

	src := image.NewRGBA(image.Rect(0, 0, 2, 1))
	lr, lg, lb := encodeTestNormal(left[0], left[1], left[2])
	rr, rg, rb := encodeTestNormal(right[0], right[1], right[2])
	copy(src.Pix[0:4], []uint8{lr, lg, lb, 0xFF})
	copy(src.Pix[4:8], []uint8{rr, rg, rb, 0xFF})

	// The decoded source vectors are what the interpolation sees.
	ldx, ldy, ldz := decodeTestNormal(lr, lg, lb)
	rdx, rdy, rdz := decodeTestNormal(rr, rg, rb)

	actual := ResizeNormalMap(src, 4, 1, Linear)

	normalize := func(x, y, z float64) (float64, float64, float64) {
		l := math.Hypot(math.Hypot(x, y), z)
		return x / l, y / l, z / l
	}

	// Linear upscaling 2->4 gives the source samples at the ends and the 0.75/0.25
	// and 0.25/0.75 weighted means in between.
	n0x, n0y, n0z := normalize(ldx, ldy, ldz)
	n3x, n3y, n3z := normalize(rdx, rdy, rdz)
	n1x, n1y, n1z := normalize(0.75*ldx+0.25*rdx, 0.75*ldy+0.25*rdy, 0.75*ldz+0.25*rdz)
	n2x, n2y, n2z := normalize(0.25*ldx+0.75*rdx, 0.25*ldy+0.75*rdy, 0.25*ldz+0.75*rdz)

	expected := [][3]float64{
		{n0x, n0y, n0z},
		{n1x, n1y, n1z},
		{n2x, n2y, n2z},
		{n3x, n3y, n3z},
	}

	for x, e := range expected {
		assertUnitNormal(t, "Linear 2->4", actual, x, 0, e[0], e[1], e[2], 0.01)
		if alpha := actual.Pix[x*4+3]; alpha != 0xFF {
			t.Errorf("Linear 2->4: pixel %d: expected alpha 255, got %d", x, alpha)
		}
	}

	// The interpolated middle directions should be a renormalized blend, not a plain
	// color average: the source directions differ, so a raw color average would fall
	// short of unit length.
	for x := 1; x < 3; x++ {
		pos := x * 4
		nx, ny, nz := decodeTestNormal(actual.Pix[pos], actual.Pix[pos+1], actual.Pix[pos+2])
		if l := math.Hypot(math.Hypot(nx, ny), nz); math.Abs(l-1) > 0.01 {
			t.Errorf("Linear 2->4: pixel %d: expected renormalized vector, length %f", x, l)
		}
	}
}

func TestResizeNormalMapDownscaleMean(t *testing.T) {
	// Analytic normal field n(x, y) = normalize(x-1.5, y-1.5, 2) sampled at pixel centres.
	src := image.NewRGBA(image.Rect(0, 0, 4, 4))
	field := func(x, y int) (uint8, uint8, uint8) {
		nx := float64(x) - 1.5
		ny := float64(y) - 1.5
		nz := 2.0
		l := math.Hypot(math.Hypot(nx, ny), nz)
		return encodeTestNormal(nx/l, ny/l, nz/l)
	}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			r, g, b := field(x, y)
			pos := y*src.Stride + x*4
			src.Pix[pos+0] = r
			src.Pix[pos+1] = g
			src.Pix[pos+2] = b
			src.Pix[pos+3] = 0xFF
		}
	}

	// Box downsampling 4x4 -> 2x2 averages each disjoint 2x2 block.
	actual := ResizeNormalMap(src, 2, 2, Box)

	for y := 0; y < 2; y++ {
		for x := 0; x < 2; x++ {
			var mx, my, mz float64
			for by := 0; by < 2; by++ {
				for bx := 0; bx < 2; bx++ {
					r, g, b := field(x*2+bx, y*2+by)
					nx, ny, nz := decodeTestNormal(r, g, b)
					mx += nx
					my += ny
					mz += nz
				}
			}
			l := math.Hypot(math.Hypot(mx, my), mz)
			assertUnitNormal(t, "Box 4->2", actual, x, y, mx/l, my/l, mz/l, 0.01)
		}
	}
}

func TestResizeNormalMapAlphaAndTransparency(t *testing.T) {
	flatR, flatG, flatB := encodeTestNormal(0, 0, 1)

	// Opaque flat normals with one fully transparent pixel carrying a valid but
	// unrelated direction, which must not influence the result.
	src := image.NewRGBA(image.Rect(0, 0, 4, 1))
	alphas := []uint8{0xFF, 0xFF, 0x00, 0xFF}
	for x := 0; x < 4; x++ {
		pos := x * 4
		if alphas[x] == 0 {
			// A strongly red encoded direction: wrong on purpose.
			src.Pix[pos+0] = 0xFF
			src.Pix[pos+1] = 0x00
			src.Pix[pos+2] = 0x00
			src.Pix[pos+3] = 0x00
			continue
		}
		src.Pix[pos+0] = flatR
		src.Pix[pos+1] = flatG
		src.Pix[pos+2] = flatB
		src.Pix[pos+3] = alphas[x]
	}

	actual := ResizeNormalMap(src, 2, 1, Linear)

	// Linear weights for 4->2 are [0.75, 0.75, 0.25, 0] and [0, 0.25, 0.75, 0.75].
	expectedAlpha := []uint8{219, 146}
	for x := 0; x < 2; x++ {
		assertUnitNormal(t, "Transparency", actual, x, 0, 0, 0, 1, 0.01)
		if a := actual.Pix[x*4+3]; a != expectedAlpha[x] {
			t.Errorf("Transparency: pixel %d: expected alpha %d, got %d", x, expectedAlpha[x], a)
		}
	}

	// Alpha is interpolated even where the direction becomes fully transparent.
	src2 := image.NewRGBA(image.Rect(0, 0, 2, 1))
	copy(src2.Pix[0:4], []uint8{flatR, flatG, flatB, 0xFF})
	copy(src2.Pix[4:8], []uint8{flatR, flatG, flatB, 0x00})

	actual2 := ResizeNormalMap(src2, 1, 1, Linear)
	if a := actual2.Pix[3]; a != 128 {
		t.Errorf("Alpha interpolation: expected alpha 128, got %d", a)
	}
	assertUnitNormal(t, "Alpha interpolation", actual2, 0, 0, 0, 0, 1, 0.01)

	// Fully transparent source: alpha 0 and the fixed flat normal fallback.
	src3 := image.NewRGBA(image.Rect(0, 0, 2, 2))
	actual3 := ResizeNormalMap(src3, 1, 1, Lanczos)
	if r, g, b, a := actual3.At(0, 0).RGBA(); a != 0 ||
		uint8(r>>8) != flatR || uint8(g>>8) != flatG || uint8(b>>8) != flatB {
		t.Errorf("All transparent: expected flat normal with alpha 0, got %v", actual3.At(0, 0))
	}

	// Nearest neighbor keeps alpha untouched and falls back for transparent samples.
	actual4 := ResizeNormalMap(src2, 2, 1, NearestNeighbor)
	if a := actual4.Pix[3]; a != 0xFF {
		t.Errorf("Nearest neighbor: expected alpha 255, got %d", a)
	}
	if a := actual4.Pix[7]; a != 0x00 {
		t.Errorf("Nearest neighbor: expected alpha 0, got %d", a)
	}
	assertUnitNormal(t, "Nearest neighbor transparent", actual4, 1, 0, 0, 0, 1, 0.01)
}

func TestResizeNormalMapZeroLength(t *testing.T) {
	// (128, 128, 128) decodes to a zero-length vector; the canonical flat normal is
	// (128, 128, 255).
	src := image.NewRGBA(image.Rect(0, 0, 3, 2))
	for y := 0; y < 2; y++ {
		for x := 0; x < 3; x++ {
			pos := y*src.Stride + x*4
			src.Pix[pos+0] = 128
			src.Pix[pos+1] = 128
			src.Pix[pos+2] = 128
			src.Pix[pos+3] = 0xFF
		}
	}

	filters := []ResampleFilter{NearestNeighbor, Box, Linear, Gaussian, CatmullRom, Lanczos}
	sizes := [][2]int{{2, 2}, {6, 4}, {1, 1}}
	for _, f := range filters {
		for _, s := range sizes {
			actual := ResizeNormalMap(src, s[0], s[1], f)
			for y := 0; y < s[1]; y++ {
				for x := 0; x < s[0]; x++ {
					pos := y*actual.Stride + x*4
					if actual.Pix[pos+0] != 128 || actual.Pix[pos+1] != 128 || actual.Pix[pos+2] != 255 {
						t.Errorf("Zero length: pixel (%d,%d): expected flat normal, got %v",
							x, y, actual.Pix[pos:pos+3])
					}
					if a := actual.Pix[pos+3]; a != 0xFF {
						t.Errorf("Zero length: pixel (%d,%d): expected alpha 255, got %d", x, y, a)
					}
				}
			}
		}
	}
}

func TestResizeNormalMapInvalidInput(t *testing.T) {
	empty := image.NewRGBA(image.Rect(0, 0, 0, 0))
	src := image.NewRGBA(image.Rect(0, 0, 2, 2))
	filled := &image.RGBA{
		Stride: 1 * 4,
		Rect:   image.Rect(0, 0, 1, 2),
		Pix:    []uint8{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF},
	}

	cases := []struct {
		name          string
		img           image.Image
		width, height int
	}{
		{"zero width", src, 0, 4},
		{"zero height", src, 4, 0},
		{"negative width", src, -1, 4},
		{"negative height", src, 4, -2},
		{"overflowing size", src, math.MaxInt, 4},
		{"overflowing size 2", src, 4, math.MaxInt},
		{"nil image", nil, 4, 4},
		{"empty image", empty, 4, 4},
		{"empty bounds struct", &image.RGBA{Stride: 0, Rect: image.Rect(0, 0, 0, 0), Pix: []uint8{}}, 2, 2},
	}
	allFilters := []ResampleFilter{NearestNeighbor, Box, Linear, Gaussian, MitchellNetravali, CatmullRom, Lanczos}

	for _, c := range cases {
		for _, f := range allFilters {
			actual := ResizeNormalMap(c.img, c.width, c.height, f)
			if !util.RGBAImageEqual(actual, empty) {
				t.Errorf("%s: expected empty image, got bounds %v", c.name, actual.Bounds())
			}
		}
	}

	// The non-empty inputs above must not be modified.
	if !util.RGBAImageEqual(filled, &image.RGBA{
		Stride: 1 * 4,
		Rect:   image.Rect(0, 0, 1, 2),
		Pix:    []uint8{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF},
	}) {
		t.Error("invalid input handling modified the source image")
	}
}

func TestResizeNormalMapNonFiniteFilter(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 4, 3))
	for y := 0; y < 3; y++ {
		for x := 0; x < 4; x++ {
			r, g, b := encodeTestNormal(float64(x)-1.5, float64(y)-1, 2)
			pos := y*src.Stride + x*4
			src.Pix[pos+0] = r
			src.Pix[pos+1] = g
			src.Pix[pos+2] = b
			src.Pix[pos+3] = 0xFF
		}
	}

	nearest := ResizeNormalMap(src, 6, 5, NearestNeighbor)

	filters := []struct {
		name   string
		filter ResampleFilter
	}{
		{"NaN weights", ResampleFilter{Support: 1, Fn: func(x float64) float64 { return math.NaN() }}},
		{"Inf weights", ResampleFilter{Support: 1, Fn: func(x float64) float64 { return math.Inf(1) }}},
		{"NaN support", ResampleFilter{Support: math.NaN(), Fn: Linear.Fn}},
		{"Inf support", ResampleFilter{Support: math.Inf(1), Fn: Linear.Fn}},
		{"nil fn", ResampleFilter{Support: 1, Fn: nil}},
		{"zero support", ResampleFilter{Support: 0, Fn: Linear.Fn}},
	}

	for _, f := range filters {
		actual := ResizeNormalMap(src, 6, 5, f.filter)
		if !util.RGBAImageEqual(actual, nearest) {
			t.Errorf("%s: expected nearest neighbor fallback, got %s", f.name, util.RGBAToString(actual))
		}
	}
}

func TestResizeNormalMapNonZeroBounds(t *testing.T) {
	field := func(x, y int) (uint8, uint8, uint8) {
		nx := float64(x) - 1.5
		ny := float64(y) - 1.5
		nz := 3.0
		l := math.Hypot(math.Hypot(nx, ny), nz)
		return encodeTestNormal(nx/l, ny/l, nz/l)
	}

	plain := image.NewRGBA(image.Rect(0, 0, 4, 4))
	offset := image.NewRGBA(image.Rect(10, 20, 14, 24))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			r, g, b := field(x, y)
			posPlain := y*plain.Stride + x*4
			plain.Pix[posPlain+0] = r
			plain.Pix[posPlain+1] = g
			plain.Pix[posPlain+2] = b
			plain.Pix[posPlain+3] = 0xFF

			posOffset := offset.PixOffset(10+x, 20+y)
			offset.Pix[posOffset+0] = r
			offset.Pix[posOffset+1] = g
			offset.Pix[posOffset+2] = b
			offset.Pix[posOffset+3] = 0xFF
		}
	}

	before := make([]uint8, len(offset.Pix))
	copy(before, offset.Pix)

	filters := []ResampleFilter{NearestNeighbor, Box, Linear, Lanczos}
	for _, f := range filters {
		expected := ResizeNormalMap(plain, 7, 5, f)
		actual := ResizeNormalMap(offset, 7, 5, f)
		if !util.RGBAImageEqual(actual, expected) {
			t.Errorf("Non-zero bounds: expected %s, got %s", util.RGBAToString(expected), util.RGBAToString(actual))
		}
	}

	if !reflect.DeepEqual(before, offset.Pix) {
		t.Error("Non-zero bounds: source image was modified")
	}
}

func TestResizeNormalMapUnitLength(t *testing.T) {
	// A diagonal analytic field with varying alpha. Every destination pixel must carry a
	// unit vector regardless of the filter and scale.
	src := image.NewRGBA(image.Rect(0, 0, 6, 5))
	for y := 0; y < 5; y++ {
		for x := 0; x < 6; x++ {
			nx := float64(x) - 2.5
			ny := float64(y) - 2
			nz := 1.5
			l := math.Hypot(math.Hypot(nx, ny), nz)
			r, g, b := encodeTestNormal(nx/l, ny/l, nz/l)
			pos := y*src.Stride + x*4
			src.Pix[pos+0] = r
			src.Pix[pos+1] = g
			src.Pix[pos+2] = b
			src.Pix[pos+3] = uint8(40 + (x*31+y*17)%215)
		}
	}

	filters := []ResampleFilter{NearestNeighbor, Box, Linear, Gaussian, MitchellNetravali, CatmullRom, Lanczos}
	sizes := [][2]int{{1, 1}, {3, 7}, {11, 9}, {6, 5}}
	for _, f := range filters {
		for _, s := range sizes {
			actual := ResizeNormalMap(src, s[0], s[1], f)
			for y := 0; y < s[1]; y++ {
				for x := 0; x < s[0]; x++ {
					pos := y*actual.Stride + x*4
					nx, ny, nz := decodeTestNormal(actual.Pix[pos], actual.Pix[pos+1], actual.Pix[pos+2])
					l := math.Hypot(math.Hypot(nx, ny), nz)
					// Quantizing each of the three channels to 8 bits can move the
					// length by at most ~1/127.5.
					if math.Abs(l-1) > 0.02 {
						t.Errorf("Unit length: size %dx%d: pixel (%d,%d): length %f (rgb: %d,%d,%d)",
							s[0], s[1], x, y, l, actual.Pix[pos], actual.Pix[pos+1], actual.Pix[pos+2])
					}
				}
			}
		}
	}
}

func TestResizeNormalMapFlatMatchesResize(t *testing.T) {
	// A flat normal map encodes (0, 0, 1) as (128, 128, 255) everywhere. Since the
	// vector field is constant, renormalization must leave the RGB channels untouched,
	// and the alpha channel must follow the regular Resize result.
	flatR, flatG, flatB := encodeTestNormal(0, 0, 1)
	src := image.NewRGBA(image.Rect(0, 0, 5, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 5; x++ {
			pos := y*src.Stride + x*4
			src.Pix[pos+0] = flatR
			src.Pix[pos+1] = flatG
			src.Pix[pos+2] = flatB
			src.Pix[pos+3] = uint8(35 + (x*53+y*29)%220)
		}
	}

	filters := []struct {
		name string
		// nonNegative filters interpolate alpha exactly like Resize up to the
		// uint8 quantization of Resize's intermediate pass.
		nonNegative bool
		filter      ResampleFilter
	}{
		{"NearestNeighbor", true, NearestNeighbor},
		{"Box", true, Box},
		{"Linear", true, Linear},
		{"Gaussian", true, Gaussian},
		// The remaining filters have negative lobes. Resize quantizes and clamps
		// its intermediate horizontal pass, which can shift the final alpha by a
		// few units; ResizeNormalMap filters in a single pass. Their alpha is
		// covered by the dedicated transparency test instead.
		{"MitchellNetravali", false, MitchellNetravali},
		{"CatmullRom", false, CatmullRom},
		{"Lanczos", false, Lanczos},
	}
	sizes := [][2]int{{3, 2}, {10, 8}, {5, 4}, {1, 1}}

	for _, f := range filters {
		for _, s := range sizes {
			normal := ResizeNormalMap(src, s[0], s[1], f.filter)
			regular := Resize(src, s[0], s[1], f.filter)
			for y := 0; y < s[1]; y++ {
				for x := 0; x < s[0]; x++ {
					pos := y*normal.Stride + x*4
					if normal.Pix[pos+0] != flatR || normal.Pix[pos+1] != flatG || normal.Pix[pos+2] != flatB {
						t.Errorf("%s %dx%d: pixel (%d,%d): expected flat normal %d,%d,%d, got %d,%d,%d",
							f.name, s[0], s[1], x, y, flatR, flatG, flatB,
							normal.Pix[pos+0], normal.Pix[pos+1], normal.Pix[pos+2])
					}
					if !f.nonNegative {
						continue
					}
					diff := int(normal.Pix[pos+3]) - int(regular.Pix[pos+3])
					if diff < -1 || diff > 1 {
						t.Errorf("%s %dx%d: pixel (%d,%d): alpha differs from Resize: %d vs %d",
							f.name, s[0], s[1], x, y, normal.Pix[pos+3], regular.Pix[pos+3])
					}
				}
			}
		}
	}
}

func BenchmarkResizeNormalMap(b *testing.B) {
	img := image.NewRGBA(image.Rect(0, 0, 1024, 1024))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i+0] = 128
		img.Pix[i+1] = 128
		img.Pix[i+2] = 255
		img.Pix[i+3] = 255
	}
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		benchResult = ResizeNormalMap(img, 512, 512, Linear)
	}
}
