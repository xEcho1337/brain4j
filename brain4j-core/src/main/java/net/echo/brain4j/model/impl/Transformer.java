package net.echo.brain4j.model.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizer.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.brain4j.transformers.DecoderGroup;
import net.echo.brain4j.transformers.TransformerEncoder;
import net.echo.brain4j.transformers.vocabulary.VocabularyMapper;
import net.echo.math4j.BrainUtils;
import net.echo.math4j.DataSet;
import net.echo.math4j.math.tensor.Tensor;

import java.io.DataInputStream;
import java.io.DataOutputStream;

public class Transformer extends Model {

    public Transformer(Layer... layers) {
        super(layers);
    }

    @Override
    public Transformer compile(LossFunction function, Optimizer optimizer) {
        return compile(WeightInit.UNIFORM_XAVIER.getFunction(), function, optimizer, new StochasticUpdater());
    }

    @Override
    public Transformer compile(Loss function, Optimizer optimizer) {
        return compile(function.getFunction(), optimizer);
    }

    @Override
    public Transformer compile(WeightInit initializer, Loss lossFunction, Optimizer optimizer, Updater updater) {
        return compile(initializer.getFunction(), lossFunction.getFunction(), optimizer, updater);
    }

    @Override
    public Transformer compile(WeightInitializer initializer, LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        super.compile(initializer, lossFunction, optimizer, updater);

        connect(initializer);

        return this;
    }

    @Override
    public EvaluationResult evaluate(DataSet<DataRow> dataSet) {
        return null;
    }

    @Override
    public void fit(DataSet<DataRow> dataSet) {
        propagation.iteration(dataSet);
    }

    @Override
    public Tensor predict(StatesCache cache, Tensor input, boolean training) {
        Tensor result = input;

        for (Layer layer : layers) {
            result = layer.forward(cache, layer, result);
        }

        return result;
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        for (Layer layer : layers) {
            stream.writeUTF(layer.getClass().getName());
            layer.serialize(stream);
        }
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        layers.clear();
        int layersSize = stream.readInt();

        for (int i = 0; i < layersSize; i++) {
            String layerClassPath = stream.readUTF();
            Layer instance = BrainUtils.newInstance(layerClassPath);

            instance.deserialize(stream);
            layers.add(instance);
        }
    }
}
