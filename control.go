/*
package nNet provides a framework for working with neural-nets. It also defines
nNet/control which allows easier manipulation of neural nets.
*/
package nNet

func (net *Net) deleteLink(from int, to int) int {
	//Deletes weighting link from *from* to *to*
	//If its successful 1 will be returned.
	//If the values are invalid 0 will be returned.
	tnode, e := net.getNode(to)
	if e != 1 {
		//Error
		return 0
	} else if _, e := net.getNode(from); e != 1 {
		//Another Error
		return 0
	}
	delete(tnode.Inids, from)
	return 1
}

func (net *Net) addLink(from int, to int) int {
	//Adds a link from *from* to *to* with a random
	//weighting.
	_, e := net.getNode(from)
	if _, f := net.getNode(to); f != 1 || e != 1 {
		return 0
	}

	tnode, _ := net.getNode(to)
	if tnode.Inids[from] != 0 {
		tnode.Inids[from] = 0.333444555666 //See explicitly which ones are added
		//Will be random at a later point
	}
}
