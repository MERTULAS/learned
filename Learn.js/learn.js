class GradientDescent{
    constructor(data, intercept=0, slope=1, learningRate=0.00001){
        this.data = data;
        this.intercept = intercept;
        this.slope = slope;
        this.learningRate = learningRate;
        this.weights = Object.keys(this.data);
        this.heights = Object.values(this.data);
        this.stepSizeIntercept = 0;
        this.stepSizeSlope = 0;
        this.interceptMinimize = 1;
    }

    derivativeSumsOfSquaredResiduals(){
        let derivativeForIntercept = 0;
        let derivativeForSlope = 0;
        for(let i = 0; i < this.weights.length; i++)
        {
            let prediction = this.intercept + this.slope * Number(this.weights[i]);
            derivativeForIntercept += -2 * (this.heights[i] - prediction);
            derivativeForSlope += -2 * Number(this.weights[i]) * (this.heights[i] - prediction);
        }

        this.stepSizeIntercept = derivativeForIntercept * this.learningRate;
        this.intercept = this.intercept - this.stepSizeIntercept;

        this.stepSizeSlope = derivativeForSlope * this.learningRate;
        this.slope = this.slope - this.stepSizeSlope;

        return derivativeForIntercept;
    }

    successRate(){
        let sumOfSquaredResiduals = 0;
        for(let i = 0; i < this.weights.length; i++){
            let prediction = this.intercept + this.slope * Number(this.weights[i]);
            sumOfSquaredResiduals += Math.pow((this.heights[i] - prediction), 2);
        }
        return sumOfSquaredResiduals;
    }

    r2Squared(firstSSR){
        return (firstSSR - this.successRate()) / firstSSR;
    }

    optimizer(){
        let firstSSR = this.successRate();
        let i = 1;
        let timeStart = Date.now();
        while (true){
            this.interceptMinimize = this.derivativeSumsOfSquaredResiduals();
            console.log(`Epoch ${i}___ Intercept:${this.intercept}, Slope:${this.slope}`);
            if(0.01 > this.interceptMinimize && this.interceptMinimize > -0.01) break;
            i++;
        }
        let timeStop = Date.now();
        console.log(this.intercept, this.slope)
        let benchMark = (timeStop - timeStart) / 1000;
        console.log(`First SSR: ${firstSSR}`);
        console.log(`Last SSR: ${this.successRate()}`);
        console.log(`R-Squared: %${100 * this.r2Squared(firstSSR)}`);
        console.log(`Completed in ${benchMark} seconds`);
    }
}
