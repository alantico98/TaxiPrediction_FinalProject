# Movie Review Sentiment Analyzer

This project implements a multi-container MLOps system that:

* Serves sentiment predictions, either **positive** or **negative**, via a FastAPI service
* Monitors model behavior and data drift in real-time using a Streamlit dashboard
* Shares prediction logs using a Docker volume, accessible to both containers
* Includes a script that uses a test file to systematically evaluate model performance via the API
* Runs on a publicly accessible AWS EC2 instance

## Prerequisites

Before running this app, make sure you have the following installed:

### 1. Python 3.13+

You can check your version with:

```bash
python --version
```

### 2. pip (Python package manager)

```bash
pip --version
```

### 3. Git

```bash
git --version
```

### 4. WSL 2 Backend (Windows users only, and if ssh'ing through a terminal)

```bash
wsl --version
```

### 5. Docker Installed (required)

```bash
docker --version
```

## How to Deploy on an EC2 Instance

Follow these steps to run the app on an EC2 instance:

1. Navigate to AWS Console
2. In AWS Console, navigate to EC2 security groups
	- Create a Security Group for both EC2 instances
		* Create a name (e.g. ec2-app-sg)
		* Add 3 inbound rules:
			1. Add rule for SSH traffic from anywhere (Type: SSH, Source: Anywhere IPv4)
			2. Add rule for FastAPI backend (Port range=8000, Source "Anywhere (0.0.0.0/0)")
			3. Add rule for Streamlit Frontend (Port range=8501, Source "Anywhere (0.0.0.0/0)")
	- Create a RDS Security Group
		* Create a name (e.g. rds-sg)
		* Add inbound rule:
			1. Add rule for PostgreSQL port (if default, will be Port range=5432)
			2. As Source, choose the EC2 Security group you created above (e.g. ec2-app-sg)
	- One to attach to both EC2 instances (e.g. ec2-app-sg)

2. Open an EC2 Page, select "Launch Instance" for Streamlit
	- Choose a name for your web server. Ex: "My Streamlit Application"
	- Under "Application and OS Images", select "Ubuntu". The default Ubuntu 24.04 LTS version is fine
	- Choose the architecture that suits your machine type (if not defaulted to the correct one already)
	- Under "Instance Type", select "t2.micro"
	- Create a key pair so that you can ssh into the EC2 instance:
		* Create a name
		* Select "RSA" and have the private key file format in ".pem"
		* Store in a secure location for use later
	- Under Newtork Settings:
		* Select existing security group
		* Select your EC2 Security group you created (e.g. ec2-app-sg)
	- Select "Launch Instance"
	- Once inside your security group, under "Inbound rules", select "Edit Inbound Rules":
		1. Add rule for FastAPI backend
			* Port range=8000, Source "Anywhere (0.0.0.0/0)"
		2. Add rule for Streamlit FrontEnd
			* Port range=8501, Source "Anywhere (0.0.0.0/0)"1. Clone the repo using bash (if on Windows, use WSL that allows for ssh cloning):
	- Similar, but now edit the "Outbond rules" to allow for PostgreSQL connectivity
		1. Add rule for both Streamlit Frontend and FastAPI backend
			* Port 5432, Source "Default (should be the same security group as your RDS Instance)"


3. Repeat step 2, but for FastAPI

2. Open an 'Aurora and RDS' page, select "Create Database"
	- From engine, select "PostgreSQL"
	- Under Templates, select "Sandbox" (this is the free version)
	- Select "Single-AZ DB instance deployment"
	- Under Settings
		* taxi-db
		* Credentials Settings
			1. Master username: Choose a login for the DB (this will be used later)
			2. Self managed
			3. Master password: Choose a password (this will be used later)
	- Under Instance Configuration
		* Standard Class is fine (can choose db.m7g.large)
	- Leave Storage Settings as is
	- Under Connectivity settings
		* Dont connect to an EC2 compute resourece. This will be done later
		* Leave DB subnet group as default
		* Leave 'default' for VPC security group
	- Under Database authentication
		* Select "Password authentication"
	- Under Monitoring
		* "Database Insights - Standard" (select "Advanced" if using longterm)
	- Once created, select your DB under "DB identifier"
		* Scroll down to 'Security group rules'
			1. Select security group with type "Inbound"
			2. Select security group ID
			3. Click checkbox under the security group rule ID
			4. Edit inbound rules
			5. Add rule: Port Range=5432, Source=<your-ec2-security-group>
	- Record your AWS RDS endpoint (e.g. taxi-db.crsvb7bvjgmt.us-east-1.rds.amazonaws.com)

4. Navigate back to EC2 page to connect to your EC2 Instances
	- If using AWS, select your instance under "Instances" and then select "Connect" via Public IP
	- If SSH'ing (using the Key Pair you generated)
	    1. ssh -i /path/to/your/key.pem ubuntu@<EC2_PUBLIC_IPv4_ADDRESS>
	    2. If you get a "WARNING: UNPROTECTED PRIVATE KEY FILE", the following should fix it
		    1. Try moving your key to your .ssh directory (ex: "mv /path/to/your/key.pem ~/.ssh")
		    2. Restrict directory $ file permissions:
            ```bash
			chmod 700 ~/.ssh
			chmod 600 ~/.ssh/key.pem
            ```
		    3. Ensure you own the file
			```bash
            chown $USER:$USER ~/.ssh/streamlit_app_key.pem"
            ```
		4. Try connecting again it should be successful
			```bash
            ssh -i ~/.ssh/key.pem ubuntu@<EC2_PUBLIC_IPv4_ADDRESS>
            ```

5. Install Docker on both servers (once)
	- If you have trouble setting up the connection this way, navigate back to AWS Aurora & RDS
	- Once there, select your DB, then scroll down to 'Connected compute resources'
	- Select 'Setup EC2 connection'
```bash
# On EC2
# Install psql
sudo apt-get update
sudo apt-get install -y postgresql-client   # (installs the default version for your Ubuntu)

# (Optional) if you want a specific major version, e.g. 16:
# sudo apt-get install -y postgresql-client-16

# Sanity check
psql --version

# Check for connectivity to your AWS RDS DB
psql "host=<rds-endpoint>.rds.amazonaws.com port=5432 dbname=postgres user=<app-user> password=<db-password> sslmode=require" -c "select now();"
```

5. Store environment variables
```bash
# Store the environment variables on both EC2 servers so Streamlit and FastAPI can access them
export DB_HOST="<rds-endpoint>.rds.amazonaws.com"
export DB_PORT=5432
export DB_NAME="taxi-db"
export DB_USER="<app-user>"
export DB_PASSWORD="<db-password>"
export AWS_REGION="us-east-1"
# If you already know your DB url
export DB_URL="<your-db-url>"
```

5. Install Docker on both servers
```bash
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release; echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Let your user run docker without sudo
sudo usermod -aG docker $USER
# re-login to pick up group membership
exit
# log back in:
ssh -i /path/to/key.pem ubuntu@EC2_PUBLIC_DNS

# Verify installation
docker version
docker compose version
git version

# Check the Docker daemon is running
sudo systemctl status docker --no-pager

# (If Docker is installed but not enabled, start and enable it)
sudo systemctl start docker
sudo systemctl enable docker
```

5. Clone the repo to both EC2 instances
```bash
git clone https://github.com/alantico98/TaxiPrediction_FinalProject.git
cd TaxiPrediction_FinalProject

# Without merging, git checkout the dev branch
git checkout dev
```

5. (Update) Start a local MLflow tracking server with registry

6. Deploy the containers in detached mode to build and run the containers
```bash
# On FastAPI EC2 instance
docker build -t sentiment-api ./api

# On Streamlit EC2 instance
docker build -t sentiment-monitor ./monitoring

# Run on FastAPI EC2
docker run -d --rm --name api \
-p 8000:8000 \
taxi-api

# Run on Streamlit EC2
docker run -d --rm --name monitor \
-p 8501:8501 \
taxi-monitor
```

7. Access the applications using your EC2 instance Public IP

    * FastAPI docs: https://<EC2_PUBLIC_IP>:8000/docs
    * Streamlit dashboard: https://<EC2_PUBLIC_IP>:8501

8. When finished, run the following to close and clean up the docker images
```bash
# Clean up FastAPI container on FastAPI EC2
docker stop api || true
docker rmi $(APP_NAME_API) $(APP_NAME_MONITORING) || true

# Clean up Streamlit container on FastAPI EC2
```
