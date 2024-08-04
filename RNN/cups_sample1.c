//
//  main.c
//  firstcups
//
//  Created by Mike Torii on 2017/07/16.
//  Copyright © 2017年 Mike. All rights reserved.
//

#include <stdio.h>
#include <cups/cups.h>

void display_options(cups_dest_t *dest)
{
    cups_option_t *pOption = dest->options;
    for(int i=0; i<dest->num_options; i++){
        if(pOption!=NULL){
            printf("%s\t%s\n", pOption->name, pOption->value);
        
            pOption++;
        }
    }
    
}

int print_dest(void *user_date, unsigned flags, cups_dest_t *dest)
{
    if(dest == NULL) return 0;
    
    if(dest->instance){
        printf("%s/%s\n", dest->name, dest->instance);
    } else {
        puts(dest->name);
    }
    display_options(dest);
    
    return 1;
}

int main(int argc, const char * argv[]) {
    // insert code here...
    printf("---Start Test CUPS API---\n");
    
    cupsEnumDests(CUPS_DEST_FLAGS_NONE, 1000, NULL, 0, 0, print_dest, NULL);
    
    printf("---End Test CUPS API---\n");

    return 0;
}

