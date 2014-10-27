package nNet

import "math/rand"
import "math"
import "time"
import "fmt"
import "bytes"
import "encoding/gob"
import "io/ioutil"
import "strings"
import "os"

type ActiveFunc func([]float64) float64
type Node struct {
	//Presumably this stores a pointer to a function.
	Activation string          //Mathematical function that maps the input values to an output
	Inids      map[int]float64 //The ids that this node accepts input from, indexed by id.
	Id         int             //Should bet noted that this id is relative to the net
	Result     float64         //Last calculated result produced by the node.
	Error      float64
}
type Net struct {
	Nodes       []Node
	Inputids    []int   //The ids that are regarded as the input nodes
	Outputids   []int   //The ids regarded as the output nodes
	Layerconfig [][]int //The ids of each node in each layer. eg. [[0,1],[2,3,4],[5,6]]
}

func Hyperbolic(results []float64) float64 {
	var sum float64 = 0.0
	for _, num := range results {
		sum += num
	}
	return (math.Pow(math.E, sum) - math.Pow(math.E, -1*sum)) / (math.Pow(math.E, sum) + math.Pow(math.E, -1*sum))
}

var Activations map[string]ActiveFunc = map[string]ActiveFunc{"hyperbolic": Hyperbolic}

func NewNet(layers []int, activation string) *Net {
	//Generates a new neural net with randomised weightings.
	rand.Seed(time.Now().Unix())
	net := new(Net)
	cnt := 0
	for idx, elm := range layers {
		net.Layerconfig = append(net.Layerconfig, make([]int, elm))
		for i := 0; i < elm; i++ {
			net.Nodes = append(net.Nodes, Node{activation, make(map[int]float64), cnt, 0.0, 0.0})
			if idx == 0 {
				//Keep inids blank
			} else {
				idSlice := net.Layerconfig[idx-1]
				for _, k := range idSlice {
					net.Nodes[cnt].Inids[k] = (rand.Float64() * 2.0) - 1.0
				}
			}

			net.Layerconfig[idx][i] = cnt
			cnt++
		}
	}
	//The approach below in order to initially assign weightings could be simplified
	cnt = 0
	net.Inputids = net.Layerconfig[0]
	net.Outputids = net.Layerconfig[len(net.Layerconfig)-1]
	return net
}
func (net *Net) getNode(id int) (*Node, int) {
	//If successful returns node and 1, if unsuccessful
	//returns 0.
	if id >= len(net.Nodes) {
		return new(Node), 0
	} else {
		return &net.Nodes[id], 1
	}
}
func (net *Net) getNodes(ids []int) ([]*Node, int) {
	//Takes a list of ids, returns list of pointers to
	//nodes and a number that shows how many nodes were found.
	var nodes [](*Node) = [](*Node){}
	sum := 0
	for _, id := range ids {
		n, e := net.getNode(id)
		if e == 1 {
			nodes = append(nodes, n)
		}
		sum += e
	}
	return nodes, sum
}
func (net *Net) serialExecute(inputs []float64) []float64 {
	//Serial implementation of an execute function
	if len(inputs) != len(net.Inputids) {
		fmt.Println("Warning!: Wrong number of inputs passed!")
	}
	for idx, _ := range net.Nodes {
		node := &net.Nodes[idx]
		if len(node.Inids) == 0 {
			node.Result = inputs[idx]
		} else {
			inputs := []float64{}
			for key, val := range node.Inids {
				tnode, _ := net.getNode(key) //temp Node
				inputs = append(inputs, tnode.Result*val)
			}
			node.Result = Activations[node.Activation](inputs)
		}
	}
	results := []float64{}
	for _, id := range net.Outputids {
		tnode, _ := net.getNode(id)
		results = append(results, tnode.Result)
	}
	return results
}
func (net *Net) writeNet(name string) {
	var data bytes.Buffer
	enc := gob.NewEncoder(&data)
	err := enc.Encode(*net)
	if err != nil {
		fmt.Println("Error encoding network!")
		panic(err)
	}
	err = ioutil.WriteFile(strings.Join([]string{"", name}, ""), data.Bytes(), 0644)
	if err != nil {
		fmt.Println("Error writing to file!")
		panic(err)
	}
	fmt.Println(strings.Join([]string{"Network saved to: ", name}, ""))
}
func (net *Net) readNet(name string) {
	data, err := os.Open(name)
	if err != nil {
		fmt.Println(strings.Join([]string{"Failed to read file: ", name}, ""))
		panic(err)
	}
	//fmt.Println(reflect.TypeOf(data))

	dec := gob.NewDecoder(data)
	err = dec.Decode(net)
	if err != nil {
		fmt.Println("Error decoding network!")
		panic(err)
	}

}
