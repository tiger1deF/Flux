from typing import Any, List, TypeVar
import asyncio
import copy

from agents.config.models import ContextConfig, TruncationType


T = TypeVar('T')
async def truncate_items(
    items: List[T],
    context_config: ContextConfig
) -> List[T]:
    """Truncate the items based on the configuration
    
    Works with any objects that implement:
    - tokens() method returning List[int]
    - truncate() method accepting max_tokens and truncation_type
    - content property returning str
    
    Returns modified copies of items when truncation is needed
    
    :param retrieved_items: List of items to truncate
    :type retrieved_items: List[T]
    :param context_config: Configuration for truncation
    :type context_config: ContextConfig
    :return: List of truncated items (copies if modified)
    :rtype: List[T]
    """
    if context_config.truncation_type == TruncationType.TOKEN_LIMIT:
        max_tokens = context_config.max_tokens
        item_lengths = await asyncio.gather(*[
            item.token_length() for item in items
        ])
        
        result_items = []
        current_tokens = 0
        
        for idx, (item, length) in enumerate(zip(items, item_lengths)):
            if current_tokens + length > max_tokens:
                remaining = max_tokens - current_tokens
                if remaining > 0:
                    truncated_item = copy.deepcopy(item)
                    truncated_item.content = await item.truncate(
                        max_tokens = remaining,
                        truncation_type = context_config.truncation_type
                    )
                    result_items.append(truncated_item)
                break
                
            result_items.append(item)
            current_tokens += length
            
        return result_items
        
    elif context_config.truncation_type == TruncationType.MESSAGE_COUNT:
        return items[-context_config.message_count:]
        
    elif context_config.truncation_type == TruncationType.TRIM_MAX:
        max_tokens = context_config.max_tokens
        message_count = context_config.message_count
        items = items[-message_count:]
        
        item_lengths = await asyncio.gather(*[
            item.token_length() for item in items
        ])
        total_tokens = sum(item_lengths)
        
        if total_tokens <= max_tokens:
            return items
            
        target_per_item = max_tokens // len(items)
        long_items_indices = [
            i for i, length in enumerate(item_lengths) 
            if length > target_per_item
        ]
        
        excess_tokens = total_tokens - max_tokens
        tokens_per_long_item = excess_tokens // (len(long_items_indices) or 1)
        
        result_items = []
        for idx, item in enumerate(items):
            if idx in long_items_indices:
                current_length = item_lengths[idx]
                target_length = min(
                    current_length - tokens_per_long_item,
                    target_per_item
                )
                
                if target_length < current_length:
                    truncated_item = copy.deepcopy(item)
                    truncated_item.content = await item.truncate(
                        max_tokens = int(target_length),
                        truncation_type = context_config.truncation_type
                    )
                    result_items.append(truncated_item)
                else:
                    result_items.append(item)
            else:
                result_items.append(item)
                
        return result_items
         
    elif context_config.truncation_type == TruncationType.SLIDING:
        max_tokens = context_config.max_tokens
        ratio = context_config.sliding_window_ratio
        
        item_lengths = await asyncio.gather(*[
            item.token_length() for item in reversed(items)
        ])
        total_items = len(items)
        
        age_factors = [(idx + 1) / total_items for idx in range(total_items)]
        keep_ratios = [ratio + ((1 - ratio) * age) for age in age_factors]
        allowed_tokens = [int(length * ratio) for length, ratio in zip(item_lengths, keep_ratios)]
        
        result_items = []
        current_tokens = 0
        
        for idx, (item, allowed) in enumerate(zip(reversed(items), allowed_tokens)):
            if current_tokens + allowed > max_tokens:
                remaining = max_tokens - current_tokens
                if remaining > 0:
                    truncated_item = copy.deepcopy(item)
                    truncated_item.content = await item.truncate(
                        max_tokens = remaining,
                        truncation_type = context_config.truncation_type
                    )
                    result_items.insert(0, truncated_item)
                break
                
            if allowed < item_lengths[idx]:
                truncated_item = copy.deepcopy(item)
                truncated_item.content = await item.truncate(
                    max_tokens = allowed,
                    truncation_type = context_config.truncation_type
                )
                result_items.insert(0, truncated_item)
            else:
                result_items.insert(0, item)
                
            current_tokens += allowed
            
        return result_items
    
    elif context_config.truncation_type == TruncationType.PRESERVE_ENDS:
        start_count = context_config.preserve_start_messages
        end_count = context_config.preserve_end_messages
        
        if len(items) <= (start_count + end_count):
            return items
            
        start_items = items[:start_count]
        end_items = items[-end_count:]
        middle_items = items[start_count:-end_count]
        
        start_lengths, end_lengths, middle_lengths = await asyncio.gather(
            asyncio.gather(*[item.token_length() for item in start_items]),
            asyncio.gather(*[item.token_length() for item in end_items]),
            asyncio.gather(*[item.token_length() for item in middle_items])
        )
        
        start_tokens = sum(start_lengths)
        end_tokens = sum(end_lengths)
        available_tokens = context_config.max_tokens - (start_tokens + end_tokens)
        
        if available_tokens <= 0:
            return start_items + end_items
            
        middle_count = len(middle_items)
        position_ratios = [idx / middle_count for idx in range(middle_count)]
        allowed_tokens = [
            int(length * (0.3 + (0.7 * ratio)))
            for length, ratio in zip(middle_lengths, position_ratios)
        ]
        
        truncated_middle = []
        current_tokens = 0
        
        for idx, (item, allowed) in enumerate(zip(middle_items, allowed_tokens)):
            if current_tokens + allowed > available_tokens:
                remaining = available_tokens - current_tokens
                if remaining > 0:
                    truncated_item = copy.deepcopy(item)
                    truncated_item.content = await item.truncate(
                        max_tokens = remaining,
                        truncation_type = context_config.truncation_type
                    )
                    truncated_middle.append(truncated_item)
                break
                
            if allowed < middle_lengths[idx]:
                truncated_item = copy.deepcopy(item)
                truncated_item.content = await item.truncate(
                    max_tokens = allowed,
                    truncation_type = context_config.truncation_type
                )
                truncated_middle.append(truncated_item)
            else:
                truncated_middle.append(item)
                
            current_tokens += allowed
            
        return start_items + truncated_middle + end_items


