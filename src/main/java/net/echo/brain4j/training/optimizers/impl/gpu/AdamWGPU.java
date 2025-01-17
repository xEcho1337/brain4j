package net.echo.brain4j.training.optimizers.impl.gpu;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.opencl.DeviceUtils;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.threading.NeuronCacheHolder;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import org.jocl.*;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import static org.jocl.CL.*;

public class AdamWGPU extends Optimizer {

    protected Synapse[] synapses;

    protected double beta1Timestep;
    protected double beta2Timestep;

    private long size;
    protected double beta1;
    protected double beta2;
    protected double epsilon;
    protected double weightDecay;
    protected int timestep = 0;

    // OpenCL-related fields
    private cl_context context;
    private cl_command_queue commandQueue;
    private cl_kernel kernel;

    private cl_mem dFirstMomentum;
    private cl_mem dSecondMomentum;
    private cl_mem dUpdates;
    private cl_mem dGradients;
    private cl_mem dWeights;

    public AdamWGPU(double learningRate) {
        this(learningRate, 0.001);
    }

    public AdamWGPU(double learningRate, double weightDecay) {
        this(learningRate, 0.9, 0.999, 1e-8, weightDecay);
    }

    public AdamWGPU(double learningRate, double beta1, double beta2, double epsilon, double weightDecay) {
        super(learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;

        initialize();
    }

    private void initialize() {
        cl_device_id device = DeviceUtils.findDevice(DeviceUtils.DeviceType.GPU);

        System.out.println("Using " + DeviceUtils.getDeviceName());

        this.context = clCreateContext(null, 1, new cl_device_id[]{device}, null, null, null);
        this.commandQueue = clCreateCommandQueueWithProperties(context, device, null, null);

        String kernelSource = loadKernelSource();

        cl_program program = clCreateProgramWithSource(context, 1, new String[]{kernelSource}, null, null);
        clBuildProgram(program, 0, null, null, null, null);

        this.kernel = clCreateKernel(program, "adam_update", null);
    }

    private String loadKernelSource() {
        try (InputStream inputStream = getClass().getClassLoader().getResourceAsStream("kernels/adamw_kernel.cl")) {
            if (inputStream == null) {
                throw new RuntimeException("Kernel file not found: kernels/adam_update_kernel.cl");
            }

            return new String(inputStream.readAllBytes());
        } catch (IOException e) {
            throw new RuntimeException("Failed to load kernel file", e);
        }
    }

    @Override
    public void postInitialize(Model model) {
        this.synapses = new Synapse[Synapse.SYNAPSE_COUNTER];

        for (Layer layer : model.getLayers()) {
            for (Synapse synapse : layer.getSynapses()) {
                synapses[synapse.getSynapseId()] = synapse;
            }
        }

        this.size = (long) Synapse.SYNAPSE_COUNTER * Sizeof.cl_double;

        this.dFirstMomentum = DeviceUtils.createBuffer(context, CL_MEM_READ_WRITE, size);
        this.dSecondMomentum = DeviceUtils.createBuffer(context, CL_MEM_READ_WRITE, size);
        this.dUpdates = DeviceUtils.createBuffer(context, CL_MEM_READ_WRITE, size);
        this.dWeights = DeviceUtils.createBuffer(context, CL_MEM_READ_ONLY, size);
        this.dGradients = DeviceUtils.createBuffer(context, CL_MEM_READ_ONLY, size);

        DeviceUtils.writeBuffer(commandQueue, dFirstMomentum, size, new double[Synapse.SYNAPSE_COUNTER]);
        DeviceUtils.writeBuffer(commandQueue, dSecondMomentum, size, new double[Synapse.SYNAPSE_COUNTER]);
    }

    @Override
    public void postIteration(NeuronCacheHolder cacheHolder, Updater updater, List<Layer> layers) {
        this.timestep++;

        this.beta1Timestep = Math.pow(beta1, timestep);
        this.beta2Timestep = Math.pow(beta2, timestep);

        double[] gradients = new double[Synapse.SYNAPSE_COUNTER];
        double[] updates = new double[Synapse.SYNAPSE_COUNTER];
        double[] weights = new double[Synapse.SYNAPSE_COUNTER];

        for (Layer layer : layers) {
            for (Synapse synapse : layer.getSynapses()) {
                int synapseId = synapse.getSynapseId();

                double delta = synapse.getOutputNeuron().getDelta(cacheHolder);
                double value = synapse.getInputNeuron().getValue(cacheHolder);

                gradients[synapseId] = delta * value;
                weights[synapseId] = synapse.getWeight();
            }
        }

        executeKernel(weights, updates, gradients);
        applyChanges(updater, updates);
    }

    @Override
    public double update(NeuronCacheHolder cacheHolder, Synapse synapse, Object... params) {
        // CPU-based update fallback
        return 0; // Not used when GPU is enabled
    }

    private void executeKernel(double[] weights, double[] updates, double[] gradients) {
        DeviceUtils.writeBuffer(commandQueue, dWeights, size, weights);
        DeviceUtils.writeBuffer(commandQueue, dGradients, size, gradients);

        // Set kernel arguments
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(dFirstMomentum));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(dSecondMomentum));
        clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(dGradients));
        clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(dUpdates));
        clSetKernelArg(kernel, 4, Sizeof.cl_mem, Pointer.to(dWeights));
        clSetKernelArg(kernel, 5, Sizeof.cl_double, DeviceUtils.to(weightDecay));
        clSetKernelArg(kernel, 6, Sizeof.cl_double, DeviceUtils.to(beta1));
        clSetKernelArg(kernel, 7, Sizeof.cl_double, DeviceUtils.to(beta2));
        clSetKernelArg(kernel, 8, Sizeof.cl_double, DeviceUtils.to(beta1Timestep));
        clSetKernelArg(kernel, 9, Sizeof.cl_double, DeviceUtils.to(beta2Timestep));
        clSetKernelArg(kernel, 10, Sizeof.cl_double, DeviceUtils.to(epsilon));
        clSetKernelArg(kernel, 11, Sizeof.cl_double, DeviceUtils.to(learningRate));
        clSetKernelArg(kernel, 12, Sizeof.cl_int, DeviceUtils.to(Synapse.SYNAPSE_COUNTER));

        long[] globalWorkSize = new long[]{(long) Synapse.SYNAPSE_COUNTER};

        // Launch kernel
        DeviceUtils.awaitAndRunKernel(commandQueue, kernel, 1, globalWorkSize);
        DeviceUtils.readBuffer(commandQueue, dUpdates, size, updates);
    }

    private void applyChanges(Updater updater, double[] updates) {
        for (int i = 0; i < updates.length; i++) {
            Synapse synapse = synapses[i];
            double update = updates[i];

            updater.acknowledgeChange(synapse, update);
        }
    }
}