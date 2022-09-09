from kafka import KafkaProducer, KafkaConsumer
# Add configuration
import ast
import json


def init_producer():
    return KafkaProducer(bootstrap_servers="localhost:9092",
                         key_serializer=lambda k: json.dumps(k).encode(),
                         value_serializer=lambda v: json.dumps(v).encode())


def init_consumer(topic, timeout):
    return KafkaConsumer(topic=topic,
                         bootstrap_servers="localhost:9092",
                         group_id=None,
                         auto_offset_reset="earliest",
                         enable_auto_comit=False,
                         consumer_timeout_ms=timeout)


def produce_record(producer, topic, data, partition):
    return producer.send(topic=topic, partition=partition, value=data)


def consume_record(consumer):
    rec_list = list()
    for rec in consumer:
        r = rec.value.decode("utf-8")
        rec_list.append(ast.literal_eval(r))
        consumer.commit()
    return rec_list
