using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace Regression
{
    class Program
    {
        static void Main(string[] args)
        {
            double[] dataX = {0.5, 2.3, 2.9}; // data X is the Feature
            double[] dataY = {1.4, 1.9, 3.2}; // data Y is the Label

            //Feature and Label Convert into Dense Array 
            var X = Vector<double>.Build.DenseOfArray(dataX);
            var y = Vector<double>.Build.DenseOfArray(dataY);
            
            // Initialize the Linear Regression Model
            LinearRegresion linearRegresionModel = new LinearRegresion(0.001, 1000);
            linearRegresionModel.fit(X, y); // Fitting The data

            var yPredict = linearRegresionModel.predict(X);
            var yPredictArr = ((MathNet.Numerics.LinearAlgebra.Double.DenseVector)yPredict).Values.ToArray();

            ScottPlot.Plot linearRegPlot = new();
            linearRegPlot.Add.ScatterPoints(dataX, dataY);
            linearRegPlot.Add.ScatterLine(dataX, yPredictArr);
            linearRegPlot.SavePng("linearRegPlot.png", 400, 300); // The Image is on the BIN
        }
    }
    
}
