# http://docs.openstack.org/developer/python-novaclient/ref/v2/servers.html
import time, os, sys, random, re
import inspect
from os import environ as env

from  novaclient import client
import keystoneclient.v3.client as ksclient
from keystoneauth1 import loading
from keystoneauth1 import session


flavor = "ssc.medium" 
private_net = "UPPMAX 2025/1-1 Internal IPv4 Network"
floating_ip_pool_name = None
floating_ip = None
image_name = "Ubuntu 22.04 - 2024.01.15"

loader = loading.get_plugin_loader('password')

auth = loader.load_from_options(auth_url=env['OS_AUTH_URL'],
                                username=env['OS_USERNAME'],
                                password=env['OS_PASSWORD'],
                                project_name=env['OS_PROJECT_NAME'],
                                project_domain_id=env['OS_PROJECT_DOMAIN_ID'],
                                #project_id=env['OS_PROJECT_ID'],
                                user_domain_name=env['OS_USER_DOMAIN_NAME'])

sess = session.Session(auth=auth)
nova = client.Client('2.1', session=sess)
print ("user authorization completed.")

image = nova.glance.find_image(image_name)

flavor = nova.flavors.find(name=flavor)

if private_net != None:
    net = nova.neutron.find_network(private_net)
    nics = [{'net-id': net.id}]
else:
    sys.exit("private-net not defined.")

#print("Path at terminal when executing this file")
#print(os.getcwd() + "\n")
cfg_file_path =  os.getcwd()+'/dev-cloud-cfg.txt'
if os.path.isfile(cfg_file_path):
    userdata_prod = open(cfg_file_path)
else:
    sys.exit("dev-cloud-cfg.txt is not in current working directory")

cfg_file_path =  os.getcwd()+'/dev-cloud-cfg.txt'
if os.path.isfile(cfg_file_path):
    userdata_dev = open(cfg_file_path)
else:
    sys.exit("dev-cloud-cfg.txt is not in current working directory")    

secgroups = ['default']

print ("Creating instances ... ")
instance_dev2 = nova.servers.create(name="G2dev_server_med2", image=image, flavor=flavor, key_name='Group2Project',userdata=userdata_prod, nics=nics,security_groups=secgroups)
instance_dev3 = nova.servers.create(name="G2dev_server_med3", image=image, flavor=flavor, key_name='Group2Project',userdata=userdata_dev, nics=nics,security_groups=secgroups)
inst_status_dev2 = instance_dev2.status
inst_status_dev3 = instance_dev3.status

print ("waiting for 10 seconds.. ")
time.sleep(10)

while inst_status_dev2 == 'BUILD' or inst_status_dev3 == 'BUILD':
    print ("Instance: "+instance_dev2.name+" is in "+inst_status_dev2+" state, sleeping for 5 seconds more...")
    print ("Instance: "+instance_dev3.name+" is in "+inst_status_dev3+" state, sleeping for 5 seconds more...")
    time.sleep(5)
    instance_dev2 = nova.servers.get(instance_dev2.id)
    inst_status_dev2 = instance_dev2.status
    instance_dev3 = nova.servers.get(instance_dev3.id)
    inst_status_dev3 = instance_dev3.status

ip_address_dev2 = None
for network in instance_dev2.networks[private_net]:
    if re.match('\d+\.\d+\.\d+\.\d+', network):
        ip_address_dev2 = network
        break
if ip_address_dev2 is None:
    raise RuntimeError('No IP address assigned!')

ip_address_dev3 = None
for network in instance_dev3.networks[private_net]:
    if re.match('\d+\.\d+\.\d+\.\d+', network):
        ip_address_dev3 = network
        break
if ip_address_dev3 is None:
    raise RuntimeError('No IP address assigned!')

print ("Instance: "+ instance_dev2.name +" is in " + inst_status_dev2 + " state" + " ip address: "+ ip_address_dev2)
print ("Instance: "+ instance_dev3.name +" is in " + inst_status_dev3 + " state" + " ip address: "+ ip_address_dev3)

