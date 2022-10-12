from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
# Add configuration
import ast
import json


def create_topic(topic_name):
    admin_client = KafkaAdminClient(bootstrap_servers="localhost:9092",
                                    client_id="test")
    topic_list = (NewTopic(name=topic_name, num_partitions=1, replication_factor=1), )
    admin_client.create_topics(new_topics=topic_list, validate_only=False)


def init_producer():
    return KafkaProducer(bootstrap_servers="localhost:9092",
                         api_version=(1, 0, 0),
                         key_serializer=lambda k: json.dumps(k).encode(),
                         value_serializer=lambda v: json.dumps(v).encode())


def init_consumer(topic, timeout=1000):
    consumer = KafkaConsumer(bootstrap_servers="localhost:9092",
                             group_id=None,
                             auto_offset_reset="earliest",
                             enable_auto_comit=False,
                             consumer_timeout_ms=timeout)
    consumer.subscribe(topic)
    return consumer


def produce_record(producer, topic, data, partition):
    """data could be a few lines"""
    for line in data:
        producer.send(topic=topic, partition=partition, value=line)
        producer.poll(1.0)
        partition += 1
    producer.flush()


def consume_record(consumer):
    rec_list = list()
    for rec in consumer:
        r = rec.value.decode("utf-8")
        rec_list.append(ast.literal_eval(r))
        consumer.commit()
    return rec_list


if __name__ == "__main__":
    topic_data = dict()
    producer1 = init_producer()
    topic1 = ("news", )
    data1 = ["Columbia University", "University of California"]
    produce_record(producer1, topic1[0], data1, 0)
    consumer1 = init_consumer(topic1, 1000)
    consumer2 = init_consumer(topic1, 1000)
    consume_record(consumer1)
    data2 = ["CNN", "NBC News"]
    produce_record(producer1, topic1[0], data2, 0)
    consume_record(consumer2)
