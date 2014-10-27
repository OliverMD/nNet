package nNet

/*
This file defines structs that can be used by the encoding/json module.
It defines conversions to and from these types.
*/
import "strconv"

type JNode struct {
	Activation string
	Inids      map[string]float64
	Id         int
	Result     float64
}
type JNet struct {
	JNodes      []JNode
	Inputids    []int
	Outputids   []int
	Layerconfig [][]int
}

func (node *Node) toJNode() (jnode JNode) {
	jnode.Activation = node.Activation
	jnode.Id = node.Id
	jnode.Result = node.Result
	jnode.Inids = make(map[string]float64)
	for id, weight := range node.Inids {
		jnode.Inids[strconv.Itoa(id)] = weight
	}
	return jnode
}

func (net *Net) toJNet() (jnet JNet) {
	jnet.Inputids = net.Inputids
	jnet.Outputids = net.Outputids
	jnet.Layerconfig = net.Layerconfig
	jnet.JNodes = make([]JNode, len(net.Nodes))
	for idx, node := range net.Nodes {
		jnet.JNodes[idx] = node.toJNode()
	}
	return jnet
}
func (jnode *JNode) toNode() (node Node) {
	node.Activation = jnode.Activation
	node.Id = jnode.Id
	node.Result = jnode.Result
	node.Inids = make(map[int]float64)
	for jid, weight := range jnode.Inids {
		id, _ := strconv.Atoi(jid)
		node.Inids[id] = weight
	}
	return node
}
func (jnet *JNet) toNet() (net Net) {
	net.Inputids = jnet.Inputids
	net.Outputids = jnet.Outputids
	net.Layerconfig = jnet.Layerconfig
	net.Nodes = make([]Node, len(jnet.JNodes))
	for idx, jnode := range jnet.JNodes {
		net.Nodes[idx] = jnode.toNode()
	}
	return net
}
