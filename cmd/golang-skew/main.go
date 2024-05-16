package main

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"sort"
	"time"

	"gocv.io/x/gocv"
	"gonum.org/v1/gonum/mat"
)

func Probabilistic(imagePath string) {
	img := gocv.IMRead(imagePath, gocv.IMReadColor)
	if img.Empty() {
		fmt.Println("Error reading image from:", imagePath)
		return
	}
	defer img.Close()

	// Thresholding
	binary := gocv.NewMat()
	defer binary.Close()
	gocv.Threshold(img, &binary, 0, 255, gocv.ThresholdBinaryInv+gocv.ThresholdOtsu)
	// Detect lines

	lines := gocv.NewMat()
	defer lines.Close()
	gocv.HoughLinesP(binary, &lines, 1, math.Pi/180, 100)

	// Calculate angles
	var angles []float64
	for i := 0; i < lines.Rows(); i++ {
		pt1 := lines.GetVeciAt(i, 0)
		pt2 := lines.GetVeciAt(i, 1)
		angle := math.Atan2(float64(pt2[1]-pt1[1]), float64(pt2[0]-pt1[0])) * 180 / math.Pi
		if math.Abs(angle) > 10 && math.Abs(angle) < 80 {
			angles = append(angles, angle)
		}
	}

	if len(angles) == 0 {
		fmt.Println("No significant lines detected for skew correction.")
		return
	}

	// Compute median angle
	sort.Float64s(angles)
	mid := len(angles) / 2
	medianAngle := angles[mid]

	// add conditional to check if the median angle is greater than 45
	// Rotate the image
	rotated := gocv.NewMat()
	defer rotated.Close()
	center := image.Pt(img.Cols()/2, img.Rows()/2)
	matrix := gocv.GetRotationMatrix2D(center, medianAngle, 1.0)
	gocv.WarpAffine(img, &rotated, matrix, image.Pt(img.Cols(), img.Rows()))

	// Save or display the rotated image
	gocv.IMWrite("./images/2.png", rotated)
	fmt.Println("Image has been corrected and saved as corrected_image.png")
}

func correctSkew2(imgPath string) {
	img := gocv.IMRead(imgPath, gocv.IMReadGrayScale)
	if img.Empty() {
		fmt.Println("Error reading image from:", imgPath)
		return
	}
	defer img.Close()

	// gray := gocv.NewMat()
	// defer gray.Close()
	// gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)
	// gocv.BitwiseNot(img, &gray)

	thresh := gocv.NewMat()
	defer thresh.Close()
	gocv.Threshold(img, &thresh, 0, 255, gocv.ThresholdBinary|gocv.ThresholdOtsu)

	w := thresh.Cols()
	h := thresh.Rows()
	center := image.Point{X: w / 2, Y: h / 2}

	delta := 1.0
	limit := 5.0
	var bestScore float64 = -1
	var bestAngle float64

	for angle := -limit; angle <= limit; angle += delta {
		matrix := gocv.GetRotationMatrix2D(center, angle, 1.0)
		score := DetermineScore(matrix, thresh, w, h)
		if score > bestScore {
			bestScore = score
			bestAngle = angle
		}
	}
	fmt.Println(bestAngle)
	M := gocv.GetRotationMatrix2D(center, bestAngle, 1.0)
	rotated := gocv.NewMat()
	defer rotated.Close()
	gocv.WarpAffineWithParams(img, &rotated, M, image.Point{X: w, Y: h}, gocv.InterpolationCubic, gocv.BorderReplicate, color.RGBA{})
	gocv.IMWrite("./images/2.png", rotated)
}
func DetermineScore(matrix gocv.Mat, thresh gocv.Mat, width int, height int) float64 {
	rotated := gocv.NewMat()
	defer rotated.Close()

	gocv.WarpAffineWithParams(thresh, &rotated, matrix, image.Point{X: width, Y: height}, gocv.InterpolationNearestNeighbor, gocv.BorderReplicate, color.RGBA{})
	histogram := make([]float64, rotated.Rows())
	for i := 0; i < rotated.Rows(); i++ {
		sum := 0.0
		for j := 0; j < rotated.Cols(); j++ {
			val := rotated.GetUCharAt(i, j)
			sum += float64(val)
		}
		histogram[i] = sum
	}
	score := 0.0
	for i := 1; i < len(histogram); i++ {
		score += math.Pow(histogram[i]-histogram[i-1], 2)
	}
	return score
}

func main() {
	timeStart := time.Now()
	// correctSkew2("./images/p1.jpg")
	// Probabilistic("./images/p1.jpg")
	correctTextSkewness("./images/p1.jpg")
	timeEnd := time.Now()
	fmt.Println("Time taken:", timeEnd.Sub(timeStart))
}

// ensureGray converts an image to grayscale if it is not already
func ensureGray(img gocv.Mat) gocv.Mat {
	gray := gocv.NewMat()
	gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)
	return gray
}

// getFFTMagnitude computes the magnitude of the FFT of an image
func getFFTMagnitude(img gocv.Mat) *mat.Dense {
	gray := ensureGray(img)
	optGray := ensureOptimalSquare(gray)

	// Ensure the image is in a floating-point format that DFT can work with
	floatMat := gocv.NewMat()
	defer floatMat.Close()
	optGray.ConvertTo(&floatMat, gocv.MatTypeCV32F) // Convert to 32-bit floating point

	// Threshold (if needed, depending on your use case)
	threshold := gocv.NewMat()
	defer threshold.Close()
	gocv.AdaptiveThreshold(optGray, &threshold, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinary, 15, -10)

	// Perform FFT
	dft := gocv.NewMat()
	defer dft.Close()
	gocv.DFT(floatMat, &dft, gocv.DftComplexOutput+gocv.DftScale)
	// Manually shift the DFT
	rows := dft.Rows() / 2
	cols := dft.Cols() / 2
	q0 := dft.Region(image.Rect(0, 0, cols, rows))
	q1 := dft.Region(image.Rect(cols, 0, cols*2, rows))
	q2 := dft.Region(image.Rect(0, rows, cols, rows*2))
	q3 := dft.Region(image.Rect(cols, rows, cols*2, rows*2))

	qx := gocv.NewMat()
	defer qx.Close()
	qy := gocv.NewMat()
	defer qy.Close()

	q0.CopyTo(&qx)
	q3.CopyTo(&q0)
	qx.CopyTo(&q3)

	q1.CopyTo(&qy)
	q2.CopyTo(&q1)
	qy.CopyTo(&q2)

	q0.Close()
	q1.Close()
	q2.Close()
	q3.Close()

	// Split the complex output from DFT into real and imaginary parts
	real := gocv.NewMat()
	defer real.Close()
	imaginary := gocv.NewMat()
	defer imaginary.Close()
	channels := gocv.Split(dft)
	real = channels[0]
	imaginary = channels[1]

	// Compute the magnitude
	magnitude := gocv.NewMat()
	gocv.Magnitude(real, imaginary, &magnitude)

	// Convert gocv.Mat to *mat.Dense for returning
	rowsMag, colsMag := magnitude.Rows(), magnitude.Cols()
	dataMag := make([]float64, rowsMag*colsMag)
	for i := 0; i < rowsMag; i++ {
		for j := 0; j < colsMag; j++ {
			dataMag[i*colsMag+j] = float64(magnitude.GetFloatAt(i, j))
		}
	}
	denseMatrix := mat.NewDense(rowsMag, colsMag, dataMag)

	return denseMatrix
}

// ensureOptimalSquare pads the image to make its dimensions optimal for DFT
func ensureOptimalSquare(img gocv.Mat) gocv.Mat {
	rows, cols := img.Rows(), img.Cols()
	nw := gocv.GetOptimalDFTSize(int(math.Max(float64(rows), float64(cols))))
	outputImage := gocv.NewMatWithSize(nw, nw, img.Type())
	gocv.CopyMakeBorder(img, &outputImage, 0, nw-rows, 0, nw-cols, gocv.BorderConstant, color.RGBA{255, 255, 255, 0})
	return outputImage
}

func getAngleRadialProjection(m *mat.Dense, angleMax float64, num int) float64 {
	rows, cols := m.Dims()
	if rows != cols {
		panic("matrix must be square")
	}
	r := rows / 2
	c := cols / 2

	if num == 0 {
		num = 20
	}

	tr := make([]float64, int(angleMax)*num*2)
	for i := range tr {
		tr[i] = (-angleMax + float64(i)*2*angleMax/float64(len(tr)-1)) * math.Pi / 180
	}

	profileArr := make([]float64, len(tr))
	copy(profileArr, tr)

	f := func(t float64) float64 {
		valInit := 0.0
		for x := 0; x < r; x++ {
			rowIndex := c + int(float64(x)*math.Cos(t))
			colIndex := c + int(-float64(x)*math.Sin(t))
			valInit += m.At(rowIndex, colIndex)
		}
		return valInit
	}

	li := make([]float64, len(profileArr))
	for i, t := range profileArr {
		li[i] = f(t)
	}

	maxIndex := maxIndex(li)
	a := tr[maxIndex] * 180 / math.Pi

	if a == -angleMax {
		return 0
	}
	return a
}

func maxIndex(values []float64) int {
	maxIdx := 0
	for i, v := range values {
		if v > values[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}

func correctTextSkewness(imgPath string) gocv.Mat {
	img := gocv.IMRead(imgPath, gocv.IMReadColor)
	if img.Empty() {
		fmt.Println("Error reading image from:", imgPath)
		return gocv.NewMat()
	}
	defer img.Close()

	h, w, _ := img.Rows(), img.Cols(), img.Channels()
	xCenter, yCenter := w/2, h/2

	rotationAngle := getSkewedAngle(img, 0, 15)
	fmt.Printf("[INFO]: Rotation angle is %f\n", rotationAngle)

	center := image.Pt(xCenter, yCenter)
	matrix := gocv.GetRotationMatrix2D(center, rotationAngle, 1.0)
	defer matrix.Close()

	rotatedImg := gocv.NewMat()

	gocv.WarpAffineWithParams(img, &rotatedImg, matrix, image.Pt(w, h), gocv.InterpolationLinear, gocv.BorderReplicate, color.RGBA{255, 255, 255, 0})
	gocv.IMWrite("./images/result.jpg", rotatedImg)
	return rotatedImg
}

// getSkewedAngle calculates the skew angle of an image
func getSkewedAngle(img gocv.Mat, verticalImageShape int, angleMax float64) float64 {
	if angleMax == 0 {
		angleMax = 15
	}

	// Resize
	if verticalImageShape != 0 {
		ratio := float64(verticalImageShape) / float64(img.Rows())
		newWidth := int(float64(img.Cols()) * ratio)
		gocv.Resize(img, &img, image.Pt(newWidth, verticalImageShape), 0, 0, gocv.InterpolationLinear)
	}

	m := getFFTMagnitude(img)
	a := getAngleRadialProjection(m, angleMax, 0)
	return a
}
