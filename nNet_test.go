package nNet

import "fmt"
import "testing"
import "math"
import "encoding/json"
import "io/ioutil"

func activation(a []float64) float64 {
	var sum float64 = 0.0
	for _, num := range a {
		sum += num
	}
	return (math.Pow(math.E, sum) - math.Pow(math.E, -1*sum)) / (math.Pow(math.E, sum) + math.Pow(math.E, -1*sum))
}

func TestNet(t *testing.T) {
	fmt.Println(" - - - TESTNET - - - -\n\n")
	fmt.Println(" - - - TEST NET END - - - - \n\n\n")
	return
}
func TestLearn(t *testing.T) {
	fmt.Println(" - - - TESTLEARN - - - -\n\n")
	net := NewNet([]int{2, 2, 1}, "hyperbolic")
	set0 := TrainingSet{[]float64{0.5, -0.5}, []float64{0.0}}
	set1 := TrainingSet{[]float64{0.2, 0.2}, []float64{0.4}}
	_, errors := net.train([]TrainingSet{set0, set1}, 1.3, 100)

	fmt.Println("\n", errors, "\n")

	fmt.Println(" - - - TESTLEARN END - - - -")
	return
}
func TestJson(t *testing.T) {
	net := NewNet([]int{2, 2, 1}, "hyperbolic")
	fmt.Println(net.serialExecute([]float64{0.4, 0.6}))
	bytes, _ := json.Marshal(net.toJNet())
	ioutil.WriteFile("jsontest.json", bytes, 0644)

	newbytes, _ := ioutil.ReadFile("jsontest.json")
	newjnet := new(JNet)
	json.Unmarshal(newbytes, newjnet)
	newnet := newjnet.toNet()
	fmt.Println(newnet.serialExecute([]float64{0.4, 0.6}))
	return
}
