package net.echo.brain4j.layer.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.utils.Vector;

import java.util.List;

/**
 * Represents a fully connected (dense) layer in a neural network.
 */
public class DenseLayer extends Layer {

    private Layer nextLayer;
    private List<Vector> weights;

    /**
     * Constructs a new DenseLayer instance.
     *
     * @param input the number of neurons (units) in this layer, which determines
     *              the layer's capacity to learn and represent data.
     * @param activation the activation function to be applied to the output
     *                   of each neuron, enabling non-linear transformations
     *                   of the input data.
     */
    public DenseLayer(int input, Activations activation) {
        super(input, activation);
    }

    @Override
    public Kernel forward(StatesCache cache, Layer lastLayer, Kernel input) {
        List<Neuron> nextNeurons = nextLayer.getNeurons();

        int inSize = neurons.size();
        int outSize = nextNeurons.size();

        Vector inputVector = new Vector(inSize);

        for (int i = 0; i < inSize; i++) {
            inputVector.set(i, neurons.get(i).getValue(cache));
        }

        for (int i = 0; i < outSize; i++) {
            double value = weights.get(i).weightedSum(inputVector);
            nextNeurons.get(i).setValue(cache, value);
        }

        nextLayer.applyFunction(cache, this);

        return null;
    }

    public void updateWeights(Layer nextLayer, Vector[] weights) {
        this.nextLayer = nextLayer;
        this.weights = List.of(weights);
    }
}