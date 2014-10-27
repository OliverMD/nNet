package nNet

import "fmt"
import "math"

type TrainingSet struct {
	inputs  []float64
	outputs []float64
}

func (net *Net) backwardsProp(inputSet []float64, outputSet []float64, learnRate float64) (error float64) {
	/*
		A function of Net, performs one weighting change based on one set of desired inputs and outputs.
		Uses BPN model to calculate weight changes.
		Doesn't return anything, simply changes the weighting on the connections between neurons.
	*/

	//Need to populate the result variables of the nodes in the net.
	results := net.serialExecute(inputSet)
	fmt.Println("\nResults: ", results, "\n")
	error = 0.0

	//This section runs the BPN for the output nodes and changes the weightings of those that connect
	//to these nodes. Requires slightly different method to subsequent layers.
	for i, id := range net.Outputids {
		node, _ := net.getNode(id) //Easier to work with, [getNode returns ptr]
		node.Error = node.Result * (1 - node.Result) * (outputSet[i] - node.Result)
		//^^^Use the result variable for calculated error^^^

		error += math.Pow(node.Result, 2)

		fmt.Println("Calculating node: ", node.Id)

		for inid, weight := range node.Inids {
			tnode, _ := net.getNode(inid) //temp node for this block
			//Use map[id] to access weighting to change it, otherwise it passes by value
			node.Inids[inid] = weight + (learnRate * node.Error * tnode.Result)
			fmt.Println("\tChanging Weighting to node: ", inid)
		}
	}

	for idx := len(net.Layerconfig) - 2; idx >= 0; idx-- {
		for _, id := range net.Layerconfig[idx] {
			//In this block, the first part of the error is calculated, followed by summing
			//the prdouct of error and weighting of each node that this node outputs to.
			//Error = result * (1 - result) * [(output1.error * output1[thisnode].weight) + ...]
			node, _ := net.getNode(id)
			fmt.Println("Calculating node: ", node.Id)
			node.Error = node.Result * (1 - node.Result)
			var sum float64 = 0.0

			for _, outnode := range net.Nodes {
				//Iterate over Layerconfig to get ids of nodes that this node outputs
				//it data to.
				sum += outnode.Result * outnode.Inids[id]
				//A call to a non existent id will yield 0 so nothing is added.
			}

			node.Result = node.Result * sum //Finalise value of the error for this node.

			for inid, weight := range node.Inids {
				//Iterate over the input connections, changing the weightings based on the BP
				//algorithm.
				tnode, _ := net.getNode(inid) //Temp node
				fmt.Println("\tChanging Weighting to node: ", inid)
				fmt.Println("\t\tOld: ", node.Inids[inid])
				node.Inids[inid] = weight + (learnRate * node.Result * tnode.Result)
				fmt.Println("\t\tCalculation: ", weight+(learnRate*node.Result*tnode.Result))
				fmt.Println("\t\tNew: ", node.Inids[inid])
			}
		}
	}
	return error
}
func (net *Net) train(trainingSets []TrainingSet, learnRate float64, epochs int) (error float64, errorHist []float64) {
	for i := 0; i < epochs; i++ {
		error = 0.0
		for _, set := range trainingSets {
			error += net.backwardsProp(set.inputs, set.outputs, learnRate)
		}
		fmt.Println("Epoch: ", i, ", Error: ", error)
		errorHist = append(errorHist, error)
	}
	return
}
