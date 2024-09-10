using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;

namespace Regression
{
    class LinearRegresion{

        public double learningRate { get; set; } 
        public double slope { get; set; }
        public double intercept { get; set; }
        public int num_Iterator { get; set; }
        
        public LinearRegresion(double iLearningRate = 0.001, int iNum_Iterator= 1000)
        {
            learningRate = iLearningRate; 
            num_Iterator = iNum_Iterator;
        }

        public void fit(Vector<double> X, Vector<double> y)
        {   
            int numItem = X.ToColumnMatrix().RowCount;
            int numFeature = X.ToColumnMatrix().ColumnCount;
            
            for (int i = 0; i < num_Iterator; i++)
             {
                // Linear regresion line formula is Y = a + bx
                var yHat = (X.Multiply(slope)).Add(intercept); // yHat = intercept + Slope(X)
                var costDiff = (yHat).Subtract(y); //Cost difference of Predicted Value and Actual Value

                // derivative with respect to the slope
                var derivativeSlope = ((X.ToColumnMatrix().Transpose() * (costDiff)).Sum())/numItem;
                // derivative with respect to the intercept
                var derivativeIntercept = (costDiff.Sum())/numItem;

                // New Value for the Slope (b) and Intercept (a)
                slope = slope - learningRate * derivativeSlope;
                intercept = intercept - learningRate * derivativeIntercept;
                
            }
         }

        public Vector<double> predict(Vector<double> X){

            return (X.Multiply(slope)).Add(intercept);
        }
    }
    
}
