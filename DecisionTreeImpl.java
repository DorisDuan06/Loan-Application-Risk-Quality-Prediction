import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.*;
import java.lang.Math;

/**
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {
  private DecTreeNode root;
  //ordered list of class labels
  private List<String> labels; 
  //ordered list of attributes
  private List<String> attributes; 
  //map to ordered discrete values taken by attributes
  private Map<String, List<String>> attributeValues; 
  //map for getting the index
  private HashMap<String,Integer> label_inv;
  private HashMap<String,Integer> attr_inv;
  
  /**
   * Answers static questions about decision trees.
   */
  DecisionTreeImpl() {
    // no code necessary, this is void purposefully
  }

  /**
   * Build a decision tree given only a training set.
   * 
   * @param train: the training set
   */
  DecisionTreeImpl(DataSet train) {

    this.labels = train.labels;
    this.attributes = train.attributes;
    this.attributeValues = train.attributeValues;
    // Learn the decision tree here
    // Get the list of instances via train.instances
    // A recursive helper function to build the tree
    //
    // this.labels contains the possible labels for an instance
    // this.attributes contains the whole set of attribute names
    // train.instances contains the list of instances
    root = buildtree(train.instances, attributes, majorityLabel(train.instances), null);
  }

  DecTreeNode buildtree(List<Instance> instances, List<String> attributes, String defaultLabel, String parentAttributeValue) {
	  DecTreeNode node = null;
	  if (instances.isEmpty())
		  return node = new DecTreeNode(defaultLabel, null, parentAttributeValue, true);
	  if (sameLabel(instances))
		  return node = new DecTreeNode(instances.get(0).label, null, parentAttributeValue, true);
	  if (attributes.isEmpty())
		  return node = new DecTreeNode(majorityLabel(instances), null, parentAttributeValue, true);
	  
	  int m, attributeLength, maxInfoGainIndex = 0;
	  attributeLength = attributes.size();
	  double maxInfoGain, InfoGain[] = new double[attributeLength];
	  for (int i = 0; i < attributeLength; i++)
		  InfoGain[i] = InfoGain(instances, attributes.get(i));
	  maxInfoGain = 0;
	  for (int i = 0; i < attributeLength; i++)
		  if (maxInfoGain < InfoGain[i]) {
			  maxInfoGain = InfoGain[i];
			  maxInfoGainIndex = i;
		  }
	  String bestAttribute = attributes.get(maxInfoGainIndex);
	  node = new DecTreeNode(majorityLabel(instances), bestAttribute, parentAttributeValue, false);
	  
	  List<String> thisAttribute = new ArrayList<String>(attributes);
	  thisAttribute.remove(maxInfoGainIndex);
	  
	  m = attributeValues.get(bestAttribute).size();
	  for (int j = 0; j < m; j++) {
		  List<Instance> inst = new ArrayList<Instance>();
		  for (int i = 0; i < instances.size(); i++)
			  if (instances.get(i).attributes.get(getAttributeIndex(bestAttribute)).equals(attributeValues.get(bestAttribute).get(j)))
				  inst.add(instances.get(i));
		  DecTreeNode child = buildtree(inst, thisAttribute, majorityLabel(instances), attributeValues.get(bestAttribute).get(j));
		  node.addChild(child);
	  }
	  return node;	    
  }

  boolean sameLabel(List<Instance> instances) {
      // helper function
      // returns if all the instances have the same label
      // labels are in instances.get(i).label
	  int i, length = instances.size(), mark = 1;
	  for (i = 0; i < length - 1; i++)
		  if (!instances.get(i).label.equals(instances.get(i + 1).label)) {
			  mark = 0;
			  break;
		  }  
	  if (mark == 1)
		  return true;
	  else
		  return false;
  }
  
  String majorityLabel(List<Instance> instances) {
      // helper function
      // returns the majority label of a list of examples
	  int i, label1Num = 0, label2Num = 0, length = instances.size();
	  String label1 = labels.get(0), label2 = labels.get(1);
	  
	  for (i = 0; i < length; i++) {
		  if (instances.get(i).label.equals(label1))
			  label1Num++;
		  else
			  label2Num++;
	  }
	  if (label1Num >= label2Num)
		  return label1;
	  else
		  return label2;
  }

  double entropy(List<Instance> instances) {
      // helper function
      // returns the Entropy of a list of examples
	  int i, total = instances.size(), length1, length2;
	  String label1 = instances.get(0).label;
	  length1 = 1;
	  length2 = 0;
	  for (i = 1; i < total; i++) {
		  if (instances.get(i).label.equals(label1))
			  length1++;
		  else
			  length2++;
	  }
      return -length1*1.0/total * Math.log((double)length1/total)/Math.log(2) - length2*1.0/total * Math.log((double)length2/total)/Math.log(2);
  }
  
  double conditionalEntropy(List<Instance> instances, String attr) {
      // helper function
      // returns the conditional entropy of a list of examples, given the attribute attr
	  int total = instances.size(), m, attributeIndex; // m is the number of different attribute values
	  String label1 = instances.get(0).label;
	  m = attributeValues.get(attr).size();
	  attributeIndex = getAttributeIndex(attr);
	  int amount[] = new int[m], length1[] = new int[m], length2[] = new int[m]; // amount is the number of instances which have that attribute value
	  double conditionalEntropy[] = new double[m], cEntropy = 0, label1Sum = 0, label2Sum = 0;
	  for (int i = 0; i < m; i++) {
		  amount[i] = 0;
		  length1[i] = 0;
		  length2[i] = 0;
	  }
	  for (int i = 0; i < total; i++)
		  for(int j = 0; j < m; j++)
			  if (instances.get(i).attributes.get(attributeIndex) == attributeValues.get(attr).get(j)) {
				  amount[j]++;
				  if (instances.get(i).label.equals(label1))
					  length1[j]++;
				  else
					  length2[j]++;
			  }
	  for (int i = 0; i < m; i++) {
		  if (length1[i] == 0)
			  label1Sum = 0;
		  else if (length1[i] != 0)
			  label1Sum = -length1[i]*1.0/amount[i] * Math.log((double)length1[i]/amount[i])/Math.log(2);
		  if (length2[i] == 0)
			  label2Sum = 0;
		  else if (length2[i] != 0)
			  label2Sum = -length2[i]*1.0/amount[i] * Math.log((double)length2[i]/amount[i])/Math.log(2);
		  conditionalEntropy[i] = amount[i]*1.0/total * (label1Sum + label2Sum);
		  cEntropy += conditionalEntropy[i];
	  }
      return cEntropy;
  }
  
  double InfoGain(List<Instance> instances, String attr) {
      // helper function
      // returns the info gain of a list of examples, given the attribute attr
      return entropy(instances) - conditionalEntropy(instances,attr);
  }
  
  @Override
  public String classify(Instance instance) {
      // The tree is already built, when this function is called
      // this.root will contain the learnt decision tree.
      // write a recursive helper function, to return the predicted label of instance
	  DecTreeNode node = root;
	  while(!node.terminal) {
		  for (int i = 0; i < node.children.size(); i++)
			  if(instance.attributes.get(getAttributeIndex(node.attribute)).equals(node.children.get(i).parentAttributeValue)) {
				  node = node.children.get(i);
				  break;
			  }
	  }
	  return node.label;
  }
  
  @Override
  public void rootInfoGain(DataSet train) {
    this.labels = train.labels;
    this.attributes = train.attributes;
    this.attributeValues = train.attributeValues;
    // Print the Info Gain for using each attribute at the root node
    // The decision tree may not exist when this function is called.
    // Calculate the info gain with each attribute on the entire training set.
    for (int i = 0; i < train.attributes.size(); i++)
    	System.out.format("%s %.5f\n", train.attributes.get(i), entropy(train.instances) - conditionalEntropy(train.instances, train.attributes.get(i)));
  }
  
  @Override
  public void printAccuracy(DataSet test) {
    // Print the accuracy on the test set.
    // The tree is already built, when this function is called
    // It calls function classify, and compare the predicted labels.
    // List of instances: test.instances 
    // getting the real label: test.instances.get(i).label
	int i, totalNum = test.instances.size(), correctNum = 0;
	for (i = 0; i < totalNum; i++)
		if (test.instances.get(i).label.equals(classify(test.instances.get(i))))
			correctNum++;
	System.out.format("%.5f\n", correctNum*1.0/totalNum);
    return;
  }

  @Override
  /**
   * Print the decision tree in the specified format
   */
  public void print() {
    printTreeNode(root, null, 0);
  }

  /**
   * Prints the subtree of the node with each line prefixed by 4 * k spaces.
   */
  public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < k; i++)
      sb.append("    ");
    String value;
    if (parent == null)
      value = "ROOT";
    else {
      int attributeValueIndex = this.getAttributeValueIndex(parent.attribute, p.parentAttributeValue);
      value = attributeValues.get(parent.attribute).get(attributeValueIndex);
    }
    sb.append(value);
    if (p.terminal) {
      sb.append(" (" + p.label + ")");
      System.out.println(sb.toString());
    }
    else {
      sb.append(" {" + p.attribute + "?}");
      System.out.println(sb.toString());
      for (DecTreeNode child : p.children)
        printTreeNode(child, p, k + 1);
    }
  }

  /**
   * Helper function to get the index of the label in labels list
   */
  private int getLabelIndex(String label) {
    if(label_inv == null) {
        this.label_inv = new HashMap<String,Integer>();
        for(int i=0; i < labels.size();i++)
            label_inv.put(labels.get(i),i);
    }
    return label_inv.get(label);
  }

  /**
   * Helper function to get the index of the attribute in attributes list
   */
  private int getAttributeIndex(String attr) {
    if(attr_inv == null) {
        this.attr_inv = new HashMap<String,Integer>();
        for(int i=0; i < attributes.size();i++)
            attr_inv.put(attributes.get(i),i);
    }
    return attr_inv.get(attr);
  }

  /**
   * Helper function to get the index of the attributeValue in the list for the attribute key in the attributeValues map
   */
  private int getAttributeValueIndex(String attr, String value) {
    for (int i = 0; i < attributeValues.get(attr).size(); i++)
      if (value.equals(attributeValues.get(attr).get(i)))
        return i;
    return -1;
  }
}