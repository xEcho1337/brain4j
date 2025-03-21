package net.echo.brain4j.loss.impl;

import net.echo.brain4j.loss.LossFunction;
import net.echo.math4j.math.vector.Vector;

public class BinaryCrossEntropy implements LossFunction {

    @Override
    public double calculate(Vector actual, Vector predicted) {
        double loss = 0.0;

        for (int i = 0; i < actual.size(); i++) {
            double p = Math.max(Math.min(predicted.get(i), 1 - 1e-15), 1e-15);
            loss += -actual.get(i) * Math.log(p) - (1 - actual.get(i)) * Math.log(1 - p);
        }

        return loss / actual.size();
    }

    @Override
    public double getDelta(double error, double derivative) {
        return error;
    }
}
