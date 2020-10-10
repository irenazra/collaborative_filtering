import numpy as np
import pandas as pd

def initialize_and_train():

    ratings = pd.read_excel("collabFilteringData.xlsx", header = None)
    #ratings = pd.read_csv("ratings_movie_data.csv",delimiter=',', encoding="utf-8-sig")
    #ratings.head()
    #ratings = ratings.drop('timestamp',1)
    #ratings = ratings.pivot(index ='userId', columns = 'movieId')
    print(ratings)

   

    # Initialize user and item vectors as a list of vectors
    # Each item contains a list [ u_i, b]
    userVectors = []
    itemVectors = []
    for i in range(ratings.shape[0]):
        userVectors.append([np.random.rand(5),np.random.rand()])
    for i in range(ratings.shape[1]):
        itemVectors.append([np.random.rand(5),np.random.rand()])



    # Setting hyperparameters
    lr = 0.1 # Learning rate
    epochs = 300

    # Making a copy of ratings df - just for ease
    predictions = ratings.copy()

    # Main loop
    for epoch in range(epochs):
        print("EPOCH : " + str(epoch))
        loss = 0 # Reset loss
        
        # Initialize empty derivative vectors for back prop 
        # Each item contains a list [u_i, b], for each item and user vector & bias
        userDVectors = []
        itemDVectors = []
        for i in range (ratings.shape[0]):
            userDVectors.append([np.zeros(5), 0])
        for i in range (ratings.shape[1]):    
            itemDVectors.append([np.zeros(5), 0])
        m = 0 # number of non zero entries
        
        print("Forward Propagation: Make predictions")
        # Predictions, iterate through every single cell in df
        for i in range(1,ratings.shape[0]):
            for j in range(1,ratings.shape[1]):
               

                # If cell NOT null, calculate the loss and make a prediction
                if not pd.isna(ratings.iloc[i][j]):
                    m +=1 
                    # Make prediction in the form u_i * v_j + b_i + c_j 
                    prediction = userVectors[i][0]@itemVectors[j][0] \
                                +(userVectors[i][1]+itemVectors[j][1]) # adding the biases
                    
                    predictions.iloc[i][j] = prediction # write prediction into our predictions df
        
        print("Backpropagation")
        # You've already finished all your predictions
        for i in range(1,ratings.shape[0]):
            for j in range(1,ratings.shape[1]):
                if not pd.isna(ratings.iloc[i][j]):
                    error = (ratings.iloc[i][j] - predictions.iloc[i][j]) # calculate error (not squared yet to aid in derivative calculation)
                    # Calculate derivates
                    dE_dy = -2*(error)
                    dy_du = itemVectors[j][0]
                    dy_dv =  userVectors[i][0]
                    
                    # Determine the change (deltas) for every vector and bias
                    userDVectors[i][0] += (dE_dy*dy_du*-lr)/m
                    userDVectors[i][1] += (dE_dy*-lr)/m
                    itemDVectors[j][0] += (dE_dy*dy_dv*-lr)/m
                    itemDVectors[j][1] += (dE_dy*-lr)/m
                    
                    # Calculate loss
                    loss += (error**2)/m
                    
        # Updating weights
        
        # Update user vectors by respective change
        for i in range(ratings.shape[0]):
            userVectors[i][0] += userDVectors[i][0]
            userVectors[i][1] += userDVectors[i][1]
            
        # Update item vectors by respective change
        for j in range(ratings.shape[1]):
            itemVectors[j][0] += itemDVectors[j][0] 
            itemVectors[j][1] += itemDVectors[j][1] 
    
        # Print epoch info
        if ((epoch+1) % 1) == 0:
            print("MSE Loss at epoch ", (epoch+1))
            print(loss)
            print()

   
    print(ratings)
    print(predictions)
    return predictions,userVectors,itemVectors

def predict_nan_values(predictions,userVectors,itemVectors) :
    for row in range (0,predictions.shape[0]):
        for col in range (0,predictions.shape[1]):
            if (pd.isnull(predictions.iloc[row,col])):
                rowVector = np.array((userVectors[row])[0])
                rowBias = np.array((userVectors[row])[1])
                colVector = np.array((itemVectors[col])[0])
                colBias = np.array((userVectors[row])[1])
                p = np.dot(rowVector,colVector)
                predictions.iloc[row,col] = p + rowBias + colBias
    print(predictions)
    predictions.to_csv(r'predictions.csv', index = False)
            
 


if __name__ == "__main__":
    predic,userVectors,itemVectors = initialize_and_train()
    predict_nan_values(predic,userVectors,itemVectors)