import numpy as np
import pandas as pd
import time

# Number of samples to generate
num_samples = 1000

# Initialize data dictionary
data = {
    'ip_len': np.random.randint(40, 1500, num_samples),  # Typical IP packet lengths
    'ip_ttl': np.random.randint(32, 255, num_samples),  # TTL values
    'ip_proto': np.random.choice([6, 17, 1], num_samples),  # TCP (6), UDP (17), ICMP (1)
    'payload_entropy': np.random.uniform(0, 8, num_samples),  # Entropy between 0 and 8 bits
    'tcp_sport': np.zeros(num_samples, dtype=int),
    'tcp_dport': np.zeros(num_samples, dtype=int),
    'tcp_flags': np.zeros(num_samples, dtype=int),
    'tcp_window': np.zeros(num_samples, dtype=int),
    'udp_sport': np.zeros(num_samples, dtype=int),
    'udp_dport': np.zeros(num_samples, dtype=int),
    'udp_len': np.zeros(num_samples, dtype=int),
    'icmp_type': np.full(num_samples, -1, dtype=int),
    'icmp_code': np.full(num_samples, -1, dtype=int),
    'timestamp': np.array([time.time() + i for i in range(num_samples)]),  # Sequential timestamps
    'label': np.random.randint(0, 2, num_samples)  # Random labels (0 or 1)
}

# Assign TCP, UDP, and ICMP features based on protocol
for i in range(num_samples):
    proto = data['ip_proto'][i]
    if proto == 6:  # TCP
        data['tcp_sport'][i] = np.random.randint(1024, 65535)
        data['tcp_dport'][i] = np.random.choice([80, 443, 22, 3389, 1024])
        data['tcp_flags'][i] = np.random.choice([2, 16, 24])  # SYN, ACK, etc.
        data['tcp_window'][i] = np.random.randint(1000, 65535)
    elif proto == 17:  # UDP
        data['udp_sport'][i] = np.random.randint(1024, 65535)
        data['udp_dport'][i] = np.random.choice([53, 123, 161])  # DNS, NTP, SNMP
        data['udp_len'][i] = np.random.randint(8, 1500)
    elif proto == 1:  # ICMP
        data['icmp_type'][i] = np.random.choice([0, 8])  # Echo reply, Echo request
        data['icmp_code'][i] = 0

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('dataset.csv', index=False)
print("Synthetic dataset saved to 'dataset.csv'")