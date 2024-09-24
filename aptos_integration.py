from aptos_sdk.account import Account
from aptos_sdk.client import RestClient
from aptos_sdk.transactions import EntryFunction, TransactionArgument, TransactionPayload
from aptos_sdk.type_tag import TypeTag, StructTag
import hashlib

class AptosIntegration:
    def __init__(self, node_url, private_key):
        self.client = RestClient(node_url)
        self.account = Account.load_key(private_key)

    def store_checkpoint(self, epoch, model_hash):
        payload = EntryFunction.natural(
            "0x1::distributed_training",
            "store_checkpoint",
            [],
            [
                TransactionArgument(epoch, Serializer.u64),
                TransactionArgument(model_hash, Serializer.string),
            ],
        )
        
        signed_transaction = self.client.create_single_signer_bcs_transaction(
            self.account, TransactionPayload(payload)
        )
        
        self.client.submit_bcs_transaction(signed_transaction)

    def verify_participation(self, participant_address):
        payload = EntryFunction.natural(
            "0x1::distributed_training",
            "verify_participation",
            [],
            [TransactionArgument(participant_address, Serializer.address)],
        )
        
        signed_transaction = self.client.create_single_signer_bcs_transaction(
            self.account, TransactionPayload(payload)
        )
        
        self.client.submit_bcs_transaction(signed_transaction)

    def distribute_rewards(self):
        payload = EntryFunction.natural(
            "0x1::distributed_training",
            "distribute_rewards",
            [],
            [],
        )
        
        signed_transaction = self.client.create_single_signer_bcs_transaction(
            self.account, TransactionPayload(payload)
        )
        
        self.client.submit_bcs_transaction(signed_transaction)

def compute_model_hash(model_state_dict):
    # Compute a hash of the model parameters
    hasher = hashlib.sha256()
    for param in model_state_dict.values():
        hasher.update(param.data.cpu().numpy().tobytes())
    return hasher.hexdigest()