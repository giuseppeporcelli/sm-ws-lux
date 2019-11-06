# Creating a Managed Notebook Instance

## Overview

Amazon SageMaker is a fully-managed service that enables developers and data scientists to quickly and easily build, train, tune and deploy machine learning models at any scale. Amazon SageMaker removes all the barriers that typically slow down developers who want to use machine learning.

In this lab, we will walk you through creating a S3 bucket for storing data and a Jupyter notebook instance managed by Amazon SageMaker.

## Create an Amazon S3 bucket
In this section, we will create an Amazon S3 bucket to be used as the storage service for this workshop. Amazon SageMaker uses **Amazon S3** as the main storage for both data and model artifacts; you can actually use other sources when loading data into the Jupyter notebook instances, but this is outside of the scope of this workshop and not a recommended approach for large-scale Machine Learning.

1. Sign into the **AWS Management Console** using the provided URL and credentials
2. In the upper-right corner of the AWS Management Console, confirm you are in the desired AWS region. For this workshop we will use the **EU West (Ireland)** [eu-west-1]
3. Open the **Amazon S3** console at <a href="https://console.aws.amazon.com/s3">https://console.aws.amazon.com/s3</a> or choose the Amazon S3 service in the menu.
4.	In the Amazon S3 console, click the **Create Bucket** button.
	![Amazon S3 create bucket](images/create_bucket.png)
5.	For the **Bucket Name**, type _**[username]**-sm-workshop-lux_ in the text box and click Next (take note of the bucket name, as it will be needed for next workshop activities). Press **Next** to move to the next screen.
Note: if the bucket name is already taken, feel free to add an extra prefix.

	![Amazon S3 create bucket window](images/create_bucket_window.png)
6. Enable versioning of the objects in the bucket as shown in the screen below. Enabling versioning allows to keep different versions of the data; although this is not strictly required for this workshop, it is still a good practice to track versions of data used to train ML models.

	![Amazon S3 create bucket window](images/s3_versioning.png)

	Press **Next** and then **Next** again leaving the settings as they are in the following screen.
7. Finally, click **Create Bucket** in the Review page.


## Create a managed Jupyter Notebook instance
In this section we will  create use an Amazon SageMaker managed Jupyter notebook instance.
An **Amazon SageMaker notebook instance** is a fully managed ML compute instance running the <a href="http://jupyter.org/">**Jupyter Notebook**</a> application and Anaconda. Amazon SageMaker manages creating the instance and related resources. 

1. In the AWS Management Console, click on Services, type “SageMaker” and press enter.
	
	![Amazon SageMaker](images/sm_1.png)
2. You’ll be placed in the Amazon SageMaker dashboard. Click on **Create notebook instance**.
	
	![Amazon SageMaker](images/sm_2.png)
3. In the **Create notebook instance** screen

	![Amazon SageMaker](images/sm_5.png)

	1. Give the Notebook Instance a name like _**[username]**-nb-instance_

	2. Choose **ml.t3.medium** as **Notebook instance type**
	3. Choose the available **IAM role** (named AmazonSageMaker-ExecutionRole-XXXXX) from the dropdown list. Notebook instances require permissions to call other services including Amazon SageMaker and Amazon S3 APIs. For the purpose of this workshop, the IAM role has been pre-configured, which is very likely to happen when the notebook instance is provisioned by another team in your organization.
	4. Keep **No VPC** selected in the **VPC** dropdown list
	5. Keep **No configuration** selected in the **Lifecycle configuration** dropdown list
	6. Keep **No Custom Encryption** selected in the **Encryption key** dropdown list
	7. Modify the **Volume size in GB** to 30, in order to have more storage available on the instance
	7. Finally, click on **Create notebook instance**

4. You will be redirected to the **Notebook instances** screen
	
	Wait until the notebook instance is status is **In Service** and then click on the **Open** button to be redirected to Jupyter
	
	![Amazon SageMaker](images/sm_4.png)
	
	![Amazon SageMaker](images/sm_6.png)

### Download workshop examples to the notebook instance

In this section we will clone the GitHub repository into the Amazon SageMaker notebook instance and access the Jupyter notebooks with the code for the labs.

1. Click on **New > Terminal** in the right-hand side of the Jupyter interface
	
	![Amazon SageMaker](images/sm_7.png)

	This will open a terminal window in the Jupyter interface
	
	![Amazon SageMaker](images/sm_8.png)

2. Execute the following commands in the terminal

	```bash
	cd SageMaker
	git clone https://github.com/giuseppeporcelli/sm-ws-lux
	```
3. When the clone operation completes, close the terminal window and return to the Jupyter landing page. The folder **sm-ws-lux** will appear automatically (if not, you can hit the **Refresh** button)

	![Amazon SageMaker](images/sm_9.png)
	
4. Browse to the folder **sm-ws-lux** and move to the second lab.
