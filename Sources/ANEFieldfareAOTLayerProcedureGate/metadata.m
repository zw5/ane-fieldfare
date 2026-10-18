#import <Foundation/Foundation.h>
#import <mach/mach.h>
#import <mach/mach_time.h>
#import <objc/message.h>
#import <sys/stat.h>
#import <dlfcn.h>

static uint64_t bytesAt(NSString *root){uint64_t total=0;NSDirectoryEnumerator *e=[NSFileManager.defaultManager enumeratorAtPath:root];
    for(__unused NSString *p in e){NSDictionary *a=[e fileAttributes];if([a.fileType isEqual:NSFileTypeRegular])total+=a.fileSize;}return total;}
static uint64_t footprint(void){task_vm_info_data_t i;mach_msg_type_number_t n=TASK_VM_INFO_COUNT;
    return task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&i,&n)==KERN_SUCCESS?i.phys_footprint:0;}
static double ms(uint64_t ticks){mach_timebase_info_data_t t;mach_timebase_info(&t);return(double)ticks*t.numer/t.denom/1e6;}
int main(int argc,const char *argv[]){@autoreleasepool{if(argc!=3)return 2;NSString *root=@(argv[1]);uint32_t expected=strtoul(argv[2],NULL,10);
    NSData *mil=[NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:@"model.mil"]];
    NSData *weights=[NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:@"weights/weight.bin"] options:NSDataReadingMappedIfSafe error:nil];
    void *fw=dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW|RTLD_LOCAL);(void)fw;
    Class dc=NSClassFromString(@"_ANEInMemoryModelDescriptor"),mc=NSClassFromString(@"_ANEInMemoryModel");
    NSDictionary *wd=weights?@{@"@model_path/weights/weight.bin":@{@"offset":@0,@"data":weights}}:@{};
    id d=mil?((id(*)(Class,SEL,id,id,id))objc_msgSend)(dc,@selector(modelWithMILText:weights:optionsPlist:),mil,wd,nil):nil;
    id m=d?((id(*)(Class,SEL,id))objc_msgSend)(mc,@selector(inMemoryModelWithDescriptor:),d):nil;NSError *error=nil;
    uint64_t before=footprint(),start=mach_absolute_time();BOOL compiled=m&&((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(m,@selector(compileWithQoS:options:error:),21,@{},&error);
    double compileMs=ms(mach_absolute_time()-start);start=mach_absolute_time();BOOL loaded=compiled&&((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(m,@selector(loadWithQoS:options:error:),21,@{},&error);
    double loadMs=ms(mach_absolute_time()-start);uint64_t after=footprint();NSDictionary *attrs=loaded?((id(*)(id,SEL))objc_msgSend)(m,@selector(modelAttributes)):nil;
    NSDictionary *ids=attrs[@"ANEFModelDescription"][@"kANEFModelProcedureNameToIDMapKey"];NSArray *status=attrs[@"NetworkStatusList"];
    BOOL valid=loaded&&ids.count==expected&&status.count==expected;NSDictionary *out=@{@"schema":@"ane-fieldfare.aot-layer-procedure-metadata.v1",
      @"status":valid?@"accepted":@"rejected",@"expected_procedures":@(expected),@"compiler_id_count":@(ids.count),@"network_status_count":@(status.count),
      @"compile_count":@(compiled?1:0),@"load_count":@(loaded?1:0),@"compile_ms":@(compileMs),@"load_ms":@(loadMs),
      @"stored_artifact_bytes":@(bytesAt(root)),@"stored_weight_blob_bytes":@(weights.length),@"phys_before":@(before),@"phys_loaded":@(after),
      @"phys_load_delta":@((int64_t)after-(int64_t)before),@"program_handle_nonzero":@(loaded&&((uint64_t(*)(id,SEL))objc_msgSend)(m,@selector(programHandle))!=0),
      @"error":error.localizedDescription?:@"none",@"claim_boundary":@"Compiler metadata and one private compile/load handle only; no requests are synthesized and no numerical execution is claimed."};
    NSData *j=[NSJSONSerialization dataWithJSONObject:out options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];fwrite(j.bytes,1,j.length,stdout);fputc('\n',stdout);
    if(loaded)((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(m,@selector(unloadWithQoS:error:),21,&error);return valid?0:1;}}
