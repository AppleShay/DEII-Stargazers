# Structure

This repository implements model serving cluster using Flask, Celery, OpenStack, and Ansible.

--- 

/openstack-client
- **OpenStack instance setup scripts**  
  - `start_instances.py`  
  - `start_instances_sml.py`  
  - `start_instances_lg.py`  
  - `start_instances_2med.py`  

- **CloudInit configuration files**  
  - `prod-cloud-cfg.txt`  
  - `dev-cloud-cfg.txt`  

- **Ansible automation**  
  - `setup_var.yml`  
  - `configuration.yml`  
  - `hosts`  

- **Git Hooks**  
  - `post-receive`
  

/githubstar
- **production_server**  
  - Flask application based frontend  
    - `app.py`  
    - `templates`  
  - Celery and RabbitMQ setup  
    - `workerA.py`  
  - Machine learning model and data  
    - `final_model.pkl`
  - Docker files  
    - `Dockerfile`  
    - `docker-compose.yml`
    - `requirements.txt` 

- **development_server**  
  - Model and dataset for training/testing  
    - `final_model.pkl`  
    - `newModel.py`  
    - `features2.parquet`  
    - `csvModel.py`  
    - `features.csv`
  
 - UPPMAX 2025_1-1-openrc.sh 
---

##  Usage Instructions

1. Log in to the client virtual machine.
2. Use `start_instances.py` to initialize the deployment environment.
3. Use Ansible for automated setup and configuration of VMs.
4. Serve the application via Flask and Celery.

---

## 📝 Notes

- `trigger.txt` and `github` folder is not used in this cluster.

