package net.echo.brain4j.training.updater.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.updater.Updater;

import java.util.List;

public class StochasticUpdater extends Updater {

    @Override
    public void postIteration(List<Layer> layers, double learningRate) {
        for (Layer layer : layers) {
            for (Neuron neuron : layer.getNeurons()) {
                double deltaBias = learningRate * neuron.getTotalDelta();

                neuron.setBias(neuron.getBias() + deltaBias);
                neuron.setTotalDelta(0);
            }
        }
    }

    @Override
    public void acknowledgeChange(Synapse synapse, double change) {
        synapse.setWeight(synapse.getWeight() + change);
    }
}
