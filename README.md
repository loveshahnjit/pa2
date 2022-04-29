

Background:

● The goal of this programming assignment is to (1) use Apache Spark to train a Machine

Learning model in parallel on multiple EC2 instances and (2) use Spark's MLib to develop and

use a Machine Learning model in the cloud and (3) how to use docker to create a container for

your machine learning model to simplify model deployment.

● Implementation: build a wine quality prediction ML model in Spark over AWS.

● Model is trained in parallel using 4 EC2 instances. Then saved and loaded into a Spark

application that will perform wine quality prediction.

○ (1) This application will run on one Ec2 instance.

○ (2) Assignment must be implemented in Java on Ubuntu Linux.

● Input for model training: 2 datasets given for your ML model

○ TrainingDataset.csv: Use this dataset to train the model in parallel on multiple EC2

instances.

○ ValidationDataset.csv: Use this dataset to validate the model and optimize its

performance (i.e., select the best values for the model parameters). •

● Input for prediction testing: TestDataset.csv.

○ We will use this file, which has a similar structure with the two datasets above, to test the

functionality and performance of your prediction application.

○ Your prediction application should take such a file as input.

This file is not shared with you, but you can use the validation dataset to make sure your

application works.

○ Output: The output of your application will be a measure of the prediction performance,

specifically the F1 score, which is available in MLlib. •

○ Model Implementation: Yo u have to develop a Spark application that uses MLlib to train

for wine quality prediction using the training dataset. Yo u will use the validation dataset

to check the performance of your trained model and to potentially tune your ML model

parameters for best performance. Yo u should start with a simple linear regression or

logistic regression model from MLlib, but you can try multiple ML models to see which

one leads to better performance. For classification models, you can use 10 classes (the

wine scores are from 1 to 10).

● A Docker container will be created for the Docker application so a prediction model can be

quickly deployed across multiple different environments.

● The model training is done in parallel on 4 EC2 instances.

● The prediction with or without Docker is done on a single EC2 instance.


# Overview:

● The application is automatically parallelized. Spark DataFrames, along with MLib, are used to make this

application which runs on an Amazon Web Services EMR cluster. The task execution is taken care of, and

HDFS is used for all file inquiries (locating files, storing training models.

Step 1: Creating an EMR cluster

Log into AWS console => Click EMR Service => Click Create Cluster => Launch Cluster => Set configurations such as

launch mode, vendor, release (emr-5.30.1), spark version (2.4.5), hardware configurations (instance type and number

of instances). We will be using 4 instances where 1 is a master and 3 are slaves.

Grab the Ec2 key pair if there, if not create one as needed

Finish with creating the cluster

# Step 2: SFTP connection to Master Node for uploads


Grab the master node public dns address as shown above, open cmd on a local pc.

Optional Step: Your .pem file might be downloaded in .ppk format, in which case it must be converted

to .pem to work. The full directions are here:

https://aws.amazon.com/premiumsupport/knowledge-center/ec2-ppk-pem-conversion/


Shown above is using PuttyGen to convert the .ppk file that AWS gives to a .pem file to proceed.


Enter the following command (shown below) to authenticate yourself to your cluster with the pem key created when

generating an ec2 key pair. The format for the master node dns address is hadoop@"master node dns address goes

here".

My master node address is "***ec2-3-83-157-50.compute-1.amazonaws.com*** ".

***$sftp -i mykeypair.pem hadoop@ec2-3-83-157-50.compute-1.amazonaws.com***


Alternatively, you can use Putty as shown above (with .ppk file authentication and accepting prompt).

Alternatively, you can also use WinSCP as shown above (Easiest in my opinion)



**Upload the TrainingDataSet.csv, the ValidationDataSet.csv, and the .jar from the github page.**

**Github page: https://github.com/loveshahnjit/programmingassignment2\_java**

**If you're using Winscp, or even Putty its very similar to WinScp(shown here):**

# Step 3: HDFS file transfer

First we need to ssh into the master node with the following command:


$ssh -i mykeypair.pem hadoop@ec2-3-83-157-50.compute-1.amazonaws.com

**NOTE: If it asks you to update, do it! $sudo yum update.**

Then we transfer files over to the HDFS from the master node. This is because the slave nodes also must have

access to them. Use the following commands shown below to put the files and verify their successful transfer over.

\1. $ls

\2. $hadoop fs -put /home/hadoop/TrainingDataset.csv /user/hadoop/TrainingDataset.csv

\3. $hadoop fs -put /home/hadoop/ValidationDataset.csv /user/hadoop/ValidationDataset.csv

\4. $hdfs dfs -ls -t -R


# Step 4: Launch the application

Launch the Apache-Spark application on the EMR cluster by executing the following command:

**$spark-submit Test.jar.**

**Part 2 - right side from above image**

As you can see above, if you navigate to the monitor section, then to the Spark dashboard.


**1st Try is at: http://ec2-3-83-157-50.compute-1.amazonaws.com:18080/**

**2nd Try is at: http://ec2-3-94-64-249.compute-1.amazonaws.com:18080/**

You can see the following files have been created: folder with trained models stored, we can verify this using the

following command:

$hdfs dfs -ls -t -R.

We need the file back on our master node: $hdfs dfs -copyFromLocal TrainingModel /home/hadoop/

Compress the file: $tar czf model.tar.gz TrainingModel

Go back to the cmd/putty session, and download it onto the local machine using: get hadoop/model.tar.gz

# Step 5: Create an Ec2 Instance

First we need to create an Ec2 instance, as shown above.

Log into the AWS console and go to EC2 => Launch instance => Select AMI as shown above.

Launch with correct authentication keypair.

# Step 6: Configure Ec2 Instance

Ssh into the Ec2 instance public dns:

**Mine is: $ssh -i "mykeypair.pem" root@ec2-3-83-157-50.compute-1.amazonaws.com**

We require Scala and Spark, so run the following commands sequentially:

$wget http://downloads.typesafe.com/scala/2.11.6/scala-2.11.6.tgz

$tar -xzvf scala-2.11.6.tgz

wget https://archive.apache.org/dist/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz

tar xzvf spark-2.4.5-bin.hadoop2.7.tgz

sudo chown -R ec2-user:ec2-user spark-2.4.5-bin-hadoop2.7

sudo ln -fs spark-2.4.5-bin-hadoop2.7 /opt/spark

Open the ~/.bashrc file using this command and add the lines below: $nano ~/.bashrc

$ export SCALA\_HOME = /home/ec2-user/scala-2.11.6

$ export PATH = $PATH:/home/ec2-user/scala-2.11.6/bin

Save it: $source ~/.bashrc

Open the ~/.bash\_profile file using this command and add the lines below: $nano ~/.bash\_profile

$ export SPARK\_HOME = /opt/spark

$ PATH=$PATH:$SPARK\_HOME/bin

$ export PATH

Save it: $source ~/.bash\_profile

# Step 7: Upload the trained model and necessary files

SFTP into the EC2 instance created in Step 5 using the following commands sequentially shown below:

**Mine is: $sftp -i ec2-A.pem root@ec2-3-83-157-50.compute-1.amazonaws.com**

$put thisone.jar, $put TestDataSet.csv, $put model.tar.gz, $tar -xzvf model.tar.gz

# Step 8: Run the application to predict

**$spark-submit new.jar.**

**Part 2 - right side from above image**

**Verify run: Same result (below)**

# Step 9: Using docker to predict/verify (easier way)

● $docker pull loveshahnjit/pa2

● $docker run -v pa2/TestDataSet.csv

● $docker run -v loveshahnjit/pa2/TestDataSet.csv

