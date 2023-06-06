# Ensemble modeling

Continuing from the datafitting tutorial, we see a few extra complications that arise when dealing with collections of models. 

I have some data and I have some models. 
I want to know which models fit the data the best.

Problem: the column names are not necesarily the same as the model symbols. 
We need a way to canonicalize them. 

To resolve this, we provide a mapping from model expressions to column names for each model.
A problem with this approach is that the mapping does not necesarily have to be the same length
This means that the data being fit can be different for each model.

This isn't necesarily a problem but can lead to confusion. 
Should we assume or assert that the data being fit is the same?

I think the answer is yes. 
