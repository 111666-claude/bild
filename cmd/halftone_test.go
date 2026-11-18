package cmd

import (
	"image/color"
	"testing"
)

func TestParseHexColor(t *testing.T) {
	cases := []struct {
		input    string
		expected color.RGBA
		wantErr  bool
	}{
		{"#000", color.RGBA{0, 0, 0, 255}, false},
		{"fff", color.RGBA{255, 255, 255, 255}, false},
		{"#f00", color.RGBA{255, 0, 0, 255}, false},
		{"#abcd", color.RGBA{170, 187, 204, 221}, false},
		{"#1a2b3c", color.RGBA{26, 43, 60, 255}, false},
		{"#1a2b3c4d", color.RGBA{26, 43, 60, 77}, false},
		{"", color.RGBA{}, true},
		{"#12", color.RGBA{}, true},
		{"#gggggg", color.RGBA{}, true},
		{"#12345", color.RGBA{}, true},
	}

	for _, c := range cases {
		got, err := parseHexColor(c.input)
		if c.wantErr {
			if err == nil {
				t.Errorf("parseHexColor(%q): expected error, got %v", c.input, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("parseHexColor(%q): unexpected error %v", c.input, err)
			continue
		}
		if got != c.expected {
			t.Errorf("parseHexColor(%q): expected %v, got %v", c.input, c.expected, got)
		}
	}
}

func TestParsePalette(t *testing.T) {
	palette, err := parsePalette("#000000, #ff0000 ,#ffffff")
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if len(palette) != 3 {
		t.Fatalf("expected 3 colors, got %d", len(palette))
	}
	expected := []color.RGBA{
		{0, 0, 0, 255},
		{255, 0, 0, 255},
		{255, 255, 255, 255},
	}
	for i, want := range expected {
		r, g, b, a := palette[i].RGBA()
		got := color.RGBA{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8), uint8(a >> 8)}
		if got != want {
			t.Errorf("palette[%d]: expected %v, got %v", i, want, got)
		}
	}

	if _, err := parsePalette(""); err == nil {
		t.Error("empty palette: expected error")
	}
	if _, err := parsePalette("#000000, notacolor"); err == nil {
		t.Error("invalid palette entry: expected error")
	}
}
