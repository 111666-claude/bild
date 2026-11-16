package cmd

import (
	"fmt"
	"image"

	"github.com/anthonynsimon/bild/deltae"
	"github.com/spf13/cobra"
)

func compareDeltaE() *cobra.Command {
	var scale float64

	var cmd = &cobra.Command{
		Use:     "compare",
		Short:   "computes the CIEDE2000 color difference between two images",
		Args:    cobra.ExactArgs(3),
		Example: "compare --scale 100 input1.jpg input2.jpg heatmap.png",
		Run: func(cmd *cobra.Command, args []string) {
			fin1 := args[0]
			fin2 := args[1]
			fout := args[2]

			apply2(fin1, fin2, fout, func(img1, img2 image.Image) (image.Image, error) {
				result, err := deltae.CompareWithScale(img1, img2, scale)
				if err != nil {
					return nil, err
				}

				fmt.Printf("mean deltaE: %.4f\n", result.Mean)
				fmt.Printf("max deltaE: %.4f\n", result.Max)

				return result.Heatmap, nil
			})
		}}

	cmd.Flags().Float64VarP(&scale, "scale", "s", deltae.DefaultScale, "deltaE value that maps to full heatmap intensity")

	return cmd
}

func createDeltaE() *cobra.Command {
	var cmd = &cobra.Command{
		Use:   "deltae",
		Short: "CIEDE2000 color difference analysis on images",
	}

	cmd.AddCommand(compareDeltaE())

	return cmd
}
