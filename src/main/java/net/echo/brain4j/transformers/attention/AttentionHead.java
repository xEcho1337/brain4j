package net.echo.brain4j.transformers.attention;

import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class AttentionHead {

    protected final int inputDimension;
    protected final int headDimension;
    protected final double temperature;

    protected final float[][] queryWeights;
    protected final float[][] keyWeights;
    protected final float[][] valueWeights;

    public AttentionHead(WeightInit weightInit, int inputDimension, int headDimension, double temperature) {
        this.inputDimension = inputDimension;
        this.headDimension = headDimension;
        this.temperature = temperature;

        this.queryWeights = new float[inputDimension][headDimension];
        this.keyWeights = new float[inputDimension][headDimension];
        this.valueWeights = new float[inputDimension][headDimension];

        initializeWeights(weightInit);
    }

    public int size() {
        int total = 0;

        total += queryWeights.length * queryWeights[0].length;
        total += keyWeights.length * keyWeights[0].length;
        total += valueWeights.length * valueWeights[0].length;

        return total;
    }

    public List<Vector> attend(List<Vector> inputs) {
        int sequenceLength = inputs.size();

        List<Vector> queries = new ArrayList<>();
        List<Vector> keys = new ArrayList<>();
        List<Vector> values = new ArrayList<>();

        for (Vector token : inputs) {
            queries.add(multiply(token, queryWeights));
            keys.add(multiply(token, keyWeights));
            values.add(multiply(token, valueWeights));
        }

        List<Vector> output = new ArrayList<>();
        double scale = Math.sqrt(headDimension);

        for (int i = 0; i < sequenceLength; i++) {
            Vector query = queries.get(i);
            List<Double> scoreList = new ArrayList<>();

            for (int j = 0; j < sequenceLength; j++) {
                double score = query.weightedSum(keys.get(j)) / scale;
                scoreList.add(score);
            }

            Vector attentionWeights = softmax(scoreList);
            Vector headOutput = new Vector(headDimension);

            for (int j = 0; j < sequenceLength; j++) {
                Vector weightedValue = values.get(j).scale(attentionWeights.get(j));
                headOutput = headOutput.add(weightedValue);
            }

            output.add(headOutput);
        }

        return output;
    }

    protected void initializeWeights(WeightInit weightInit) {
        Random rng = new Random();
        WeightInitializer initializer = weightInit.getInitializer();

        double bound = initializer.getBound(inputDimension, headDimension);

        for (int i = 0; i < inputDimension; i++) {
            for (int j = 0; j < headDimension; j++) {
                queryWeights[i][j] = (float) (rng.nextDouble(2 * bound) - bound);
                keyWeights[i][j] = (float) (rng.nextDouble(2 * bound) - bound);
                valueWeights[i][j] = (float) (rng.nextDouble(2 * bound) - bound);
            }
        }
    }

    protected Vector multiply(Vector vector, float[][] weights) {
        Vector result = new Vector(headDimension);

        for (int j = 0; j < headDimension; j++) {
            double sum = 0.0;

            for (int i = 0; i < inputDimension; i++) {
                sum += vector.get(i) * weights[i][j];
            }

            result.set(j, sum);
        }
        return result;
    }

    protected Vector softmax(List<Double> scores) {
        Vector result = new Vector(scores.size());
        double maxScore = Double.NEGATIVE_INFINITY;

        for (double score : scores) {
            if (score > maxScore) {
                maxScore = score;
            }
        }

        double sum = 0.0;

        for (int i = 0; i < scores.size(); i++) {
            double expVal = Math.exp((scores.get(i) - maxScore) / temperature);
            result.set(i, expVal);

            sum += expVal;
        }

        for (int i = 0; i < result.size(); i++) {
            result.set(i, result.get(i) / sum);
        }

        return result;
    }
}
