package moa.classifiers.mthesis;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.mthesis.kstar.KStarNominalAttribute;
import moa.classifiers.mthesis.kstar.KStarNumericAttribute;
import moa.core.Measurement;
import weka.classifiers.lazy.kstar.KStarCache;

import weka.core.Attribute;
import weka.core.Utils;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Random;

import static weka.classifiers.lazy.kstar.KStarConstants.*;


public class KStar extends AbstractClassifier implements MultiClassClassifier {
    static final long serialVersionUID = 563421789065L;

    private Instances trainingInstances;

    private int instancesNum;

    protected int[][] m_RandClassCols;


    /**
     * Flag turning on and off the computation of random class colomns
     */
    protected int m_ComputeRandomCols = ON;

    /**
     * Flag turning on and off the initialisation of config variables
     */
    protected int m_InitFlag = ON;

    /**
     * A custom data structure for caching distinct attribute values
     * and their scale factor or stop parameter.
     */
    protected KStarCache[] cache;

    /**
     * missing value treatment
     */
    protected int m_MissingMode = M_AVERAGE;

    /**
     * 0 = use specified blend, 1 = entropic blend setting
     */
    protected int m_BlendMethod = B_SPHERE;

    /**
     * default sphere of influence blend setting
     */
    protected int m_GlobalBlend = 20;

    int C = 0;


    public IntOption nOption = new IntOption( "n", 'n', "Number of training instances", 10, 1, Integer.MAX_VALUE);

    @Override
    public double[] getVotesForInstance(Instance instance) {
        double transProb = 0.0, temp = 0.0;
        double[] classProbability = new double[instance.numClasses()];
        double[] predictedValue = new double[1];

        predictedValue[0] = 0.0;
        if (m_InitFlag == ON) {
            // need to compute them only once and will be used for all instances.
            // We are doing this because the evaluation module controls the calls.
            if (m_BlendMethod == B_ENTROPY) {
                generateRandomClassColomns();
            }
            cache = new KStarCache[instance.numAttributes()];
            for (int i = 0; i < instance.numAttributes(); i++) {
                cache[i] = new KStarCache();
            }
            m_InitFlag = OFF;
            //      System.out.println("Computing...");
        }
        // init done.
        Instance trainInstance;
        Enumeration<Instance> enu =  trainingInstances.getEnumeration();
        while (enu.hasMoreElements()) {
            trainInstance =  enu.nextElement();
            transProb = instanceTransformationProbability(instance, trainInstance);
            if(instance.classAttribute().isNominal()){
                classProbability[(int) trainInstance.classValue()] += transProb;
            }
            if(instance.classAttribute().isNumeric()){
                predictedValue[0] += transProb * trainInstance.classValue();
                temp += transProb;
            }

        }
        if(instance.classAttribute().isNominal()){
            double sum = Utils.sum(classProbability);
            if (sum <= 0.0)
                for (int i = 0; i < classProbability.length; i++)
                    classProbability[i] = (double) 1 / (double) instance.numClasses();
            else Utils.normalize(classProbability, sum);
            return classProbability;
        }
        if(instance.classAttribute().isNumeric()){
            predictedValue[0] = (temp != 0) ? predictedValue[0] / temp : 0.0;
            return predictedValue;
        }
        return new double[0];
    }

    private void generateRandomClassColomns() {

        Random generator = new Random(42);
        //    Random generator = new Random();
        m_RandClassCols = new int[NUM_RAND_COLS + 1][];
        int[] classvals = classValues();
        for (int i = 0; i < NUM_RAND_COLS; i++) {
            // generate a randomized version of the class colomn
            m_RandClassCols[i] = randomize(classvals, generator);
        }
        // original colomn is preserved in colomn NUM_RAND_COLS
        m_RandClassCols[NUM_RAND_COLS] = classvals;
    }

    private int[] classValues() {
        int[] classval = new int[instancesNum];
        for (int i = 0; i < instancesNum; i++) {
            try {
                classval[i] = (int) trainingInstances.instance(i).classValue();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        return classval;
    }

    /**
     * Returns a copy of the array with its elements randomly redistributed.
     *
     * @param array     the array to randomize.
     * @param generator the random number generator to use
     * @return a copy of the array with its elements randomly redistributed.
     */
    private int[] randomize(int[] array, Random generator) {

        int index;
        int temp;
        int[] newArray = new int[array.length];
        System.arraycopy(array, 0, newArray, 0, array.length);
        for (int j = newArray.length - 1; j > 0; j--) {
            index = (int) (generator.nextDouble() * (double) j);
            temp = newArray[j];
            newArray[j] = newArray[index];
            newArray[index] = temp;
        }
        return newArray;
    }

    @Override
    public void resetLearningImpl() {

    }

    @Override
    public void setModelContext(InstancesHeader context) {
            try {
                instancesNum = nOption.getValue();
                this.trainingInstances = new Instances(context,0); //new StringReader(context.toString())
                this.trainingInstances.setClassIndex(context.classIndex());
            } catch(Exception e) {
                System.err.println("Error: no Model Context available.");
                e.printStackTrace();
                System.exit(1);
            }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (inst.classValue() > C)
            C = (int)inst.classValue();
        if (this.trainingInstances == null) {
            this.trainingInstances = new Instances(inst.dataset());
        }
        if (this.trainingInstances.size() >= nOption.getValue()) {
            this.trainingInstances.delete(0);
        }
        this.trainingInstances.add(inst);
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[0];
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {

    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    /**
     * Calculate the probability of the first instance transforming into the
     * second instance:
     * the probability is the product of the transformation probabilities of
     * the attributes normilized over the number of instances used.
     *
     * @param first  the test instance
     * @param second the train instance
     * @return transformation probability value
     */
    private double instanceTransformationProbability(Instance first, Instance second) {
        double transProb = 1.0;
        int numAttributes = first.numAttributes();
        int numMissAttr = 0;
        for (int i = 0; i < numAttributes; i++) {
            if (i == trainingInstances.classIndex()) {
                continue; // ignore class attribute
            }
            if (first.isMissing(i)) { // test instance attribute value is missing
                numMissAttr++;
                continue;
            }
            transProb *= attrTransProb(first, second, i);
            // normilize for missing values
            if (numMissAttr != numAttributes) {
                transProb = Math.pow(transProb, (double) numAttributes /
                        (numAttributes - numMissAttr));
            } else { // weird case!
                transProb = 0.0;
            }
        }
        // normilize for the train dataset
        return transProb / trainingInstances.size();
    }
    /**
     * Calculates the transformation probability of the indexed test attribute
     * to the indexed train attribute.
     *
     * @param first  the test instance.
     * @param second the train instance.
     * @param col    the index of the attribute in the instance.
     * @return the value of the transformation probability.
     */
    private double attrTransProb(Instance first, Instance second, int col) {

        double transProb = 0.0;
        KStarNominalAttribute ksNominalAttr;
        KStarNumericAttribute ksNumericAttr;
        if(trainingInstances.attribute(col).isNumeric()){
            ksNumericAttr = new KStarNumericAttribute(first, second, col, trainingInstances, m_RandClassCols, cache[col]);
            ksNumericAttr.setOptions(m_MissingMode, m_BlendMethod, m_GlobalBlend);
            transProb = ksNumericAttr.transProb();
        }
        if(trainingInstances.attribute(col).isNominal()) {
            ksNominalAttr = new KStarNominalAttribute(first, second, col, trainingInstances, m_RandClassCols, cache[col]);
            ksNominalAttr.setOptions(m_MissingMode, m_BlendMethod, m_GlobalBlend);
            transProb = ksNominalAttr.transProb();
        }

        return transProb;
    }


}
