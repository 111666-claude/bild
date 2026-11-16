package cmd

import (
	"image"

	"github.com/anthonynsimon/bild/guidedfilter"
	"github.com/spf13/cobra"
)

func guided() *cobra.Command {
	var radius int
	var epsilon float64

	var cmd = &cobra.Command{
		Use:     "guidedfilter",
		Short:   "apply an edge-preserving guided filter using a guide image",
		Args:    cobra.ExactArgs(3),
		Example: "guidedfilter --radius 4 --epsilon 0.01 input.jpg guide.jpg output.jpg",
		Run: func(cmd *cobra.Command, args []string) {
			fin := args[0]
			fguide := args[1]
			fout := args[2]

			apply2(fin, fguide, fout, func(img, guide image.Image) (image.Image, error) {
				return guidedfilter.Filter(img, guide, radius, epsilon), nil
			})
		}}

	cmd.Flags().IntVarP(&radius, "radius", "r", 1, "the filter window's radius")
	cmd.Flags().Float64VarP(&epsilon, "epsilon", "e", 0.01, "regularization term controlling edge preservation")

	return cmd
}
